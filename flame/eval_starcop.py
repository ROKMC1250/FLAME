"""Evaluate FLAME on the STARCOP test split.

Protocol:
    * one full-tile forward pass per test tile (zero-padded to 512 when
      smaller; padding excluded via the valid mask);
    * binary mask = sigmoid(logits) > threshold (0.5) followed by a 3x3 cross
      morphological opening;
    * pixel metrics (F1 / precision / recall / IoU) from a global confusion
      over the valid pixels of all test tiles; AUPRC threshold-free;
    * emission-rate subsets: ``strong`` = plume tiles with qplume > 1000 kg/h,
      ``weak`` = the remaining plume tiles. Each subset additionally includes
      every plume-free tile, so the background false-positive load is identical
      across subsets;
    * tile-level FPR: fraction of plume-free tiles with at least one predicted
      positive pixel after the opening.

Per-seed rows are aggregated to mean +/- std over ``logs/<uid>/seed_*``.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from flame.datasets.starcop import load_tile_for_inference
from flame.eval_common import discover_runs, load_run
from flame.metrics import METRIC_KEYS, metrics_from_arrays
from flame.utils import morph_open

SUBSETS = ['all', 'strong', 'weak']
STRONG_KGH = 1000.0


def _load_test_index(root: str, csv_name: str = 'test.csv'):
    """Return [(tile_id, qplume, has_plume), ...] for test tiles present on disk."""
    df = pd.read_csv(os.path.join(root, csv_name))
    rows = []
    for _, r in df.iterrows():
        tid = str(r['id'])
        if os.path.isdir(os.path.join(root, tid)):
            rows.append((tid, float(r.get('qplume', 0.0) or 0.0),
                         bool(r.get('has_plume', False))))
    return rows


@torch.no_grad()
def evaluate_run(run: Dict, root: str, tiles, device: str,
                 threshold: float, npy_dir=None) -> Dict:
    model, cfg, _ = load_run(run, device)
    dcfg = cfg.get('data', {})
    npy_dir = npy_dir if npy_dir is not None else dcfg.get('npy_dir')
    tile_size = dcfg.get('tile_size', 512)

    buf = {s: {'pr': [], 'lb': [], 'sc': []} for s in SUBSETS}
    n_noplume = 0
    n_tile_fp = 0

    t0 = time.time()
    for tid, qplume, has_plume in tiles:
        tile_path = os.path.join(root, tid)
        cube, rgb, gt, (H, W) = load_tile_for_inference(
            tile_path, npy_dir=npy_dir, load_rgb=True, pad_to=tile_size)
        cube_b = cube.unsqueeze(0).to(device).float()
        rgb_b = rgb.unsqueeze(0).to(device).float()

        logits, _ = model(cube_b, rgb_b)
        prob = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()
        valid = (cube.abs().sum(dim=0)[:H, :W] > 0).numpy()
        gtm = (gt[:H, :W].numpy() > 0.5) & valid
        pred = morph_open(prob > threshold) & valid

        if not gtm.any():
            n_noplume += 1
            if pred.any():
                n_tile_fp += 1

        pr = pred[valid].astype(np.bool_)
        lb = gtm[valid].astype(np.bool_)
        sc = prob[valid].astype(np.float16)

        if not has_plume or not gtm.any():
            targets = SUBSETS                       # background load in every subset
        elif qplume > STRONG_KGH:
            targets = ['all', 'strong']
        else:
            targets = ['all', 'weak']
        for s in targets:
            buf[s]['pr'].append(pr)
            buf[s]['lb'].append(lb)
            buf[s]['sc'].append(sc)
    elapsed = time.time() - t0

    row = dict(seed=run['seed'], n_tiles=len(tiles), elapsed_s=elapsed,
               tile_fpr=n_tile_fp / max(n_noplume, 1))
    for s in SUBSETS:
        m = metrics_from_arrays(
            np.concatenate(buf[s]['pr']) if buf[s]['pr'] else np.empty(0, np.bool_),
            np.concatenate(buf[s]['lb']) if buf[s]['lb'] else np.empty(0, np.bool_),
            np.concatenate(buf[s]['sc']) if buf[s]['sc'] else np.empty(0, np.float16))
        for k in METRIC_KEYS:
            row[f'{s}__{k}'] = m[k]
    return row


def aggregate(rows: List[Dict], uid: str) -> Dict:
    out = {'uid': uid, 'N': len(rows)}
    cols = [f'{s}__{k}' for s in SUBSETS for k in METRIC_KEYS] + ['tile_fpr']
    for col in cols:
        vals = np.asarray([r[col] for r in rows], dtype=np.float64)
        out[col] = float(np.nanmean(vals))
        out[f'{col}__std'] = float(np.nanstd(vals))
    return out


def write_outputs(per_seed, aggregated, out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(aggregated).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(out_dir, 'metrics_per_seed.csv'), index=False)
    md_path = os.path.join(out_dir, 'metrics.md')
    with open(md_path, 'w') as f:
        f.write('# FLAME — STARCOP test metrics\n\n')
        f.write(f'Protocol: full-tile inference, sigmoid > {threshold} + 3x3 cross '
                f'morphological opening, global pixel confusion over valid pixels. '
                f'strong = qplume > {STRONG_KGH:.0f} kg/h.\n\n')
        f.write('| uid | N | F1 | Precision | Recall | IoU | AUPRC | F1-strong | F1-weak | Tile FPR |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|\n')
        for row in aggregated:
            def fmt(col):
                return f'{row[col]:.4f} ± {row[f"{col}__std"]:.4f}'
            f.write(f'| {row["uid"]} | {row["N"]} | {fmt("all__f1")} | '
                    f'{fmt("all__precision")} | {fmt("all__recall")} | '
                    f'{fmt("all__iou")} | {fmt("all__auprc")} | '
                    f'{fmt("strong__f1")} | {fmt("weak__f1")} | {fmt("tile_fpr")} |\n')
    print(f'Wrote {os.path.join(out_dir, "metrics.csv")}\nWrote {md_path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='FLAME STARCOP evaluation')
    ap.add_argument('--uid', action='append', default=None,
                    help='uid in logs/. Repeatable. Default: flame_starcop')
    ap.add_argument('--root', default='datasets/starcop',
                    help='STARCOP dataset root (contains test.csv and tile dirs)')
    ap.add_argument('--npy-dir', default=None,
                    help='optional SWIR npy cache (overrides frozen config)')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out-dir', default='results/starcop')
    ap.add_argument('--max-test-tiles', type=int, default=None)
    args = ap.parse_args(argv)

    uids = args.uid or ['flame_starcop']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tiles = _load_test_index(args.root)
    if args.max_test_tiles:
        tiles = tiles[:args.max_test_tiles]
    n_plume = sum(1 for _, _, hp in tiles if hp)
    print(f'Test tiles: {len(tiles)} ({n_plume} plume / {len(tiles) - n_plume} plume-free)')

    all_rows, aggs = [], []
    for uid in uids:
        runs = discover_runs(uid)
        if not runs:
            print(f'[skip] {uid}: no seeds with best.pt under logs/{uid}/seed_*')
            continue
        per_seed = []
        for run in runs:
            print(f'[{uid}] seed_{run["seed"]} ...')
            row = evaluate_run(run, args.root, tiles, device,
                               threshold=args.threshold, npy_dir=args.npy_dir)
            row['uid'] = uid
            per_seed.append(row)
            all_rows.append(row)
            print(f'  F1={row["all__f1"]:.4f}  P={row["all__precision"]:.4f}  '
                  f'R={row["all__recall"]:.4f}  IoU={row["all__iou"]:.4f}  '
                  f'TileFPR={row["tile_fpr"]:.4f}  ({row["elapsed_s"]:.1f}s)')
        aggs.append(aggregate(per_seed, uid))

    if not aggs:
        print('No runs evaluated.')
        return
    write_outputs(all_rows, aggs, args.out_dir, args.threshold)


if __name__ == '__main__':
    main()
