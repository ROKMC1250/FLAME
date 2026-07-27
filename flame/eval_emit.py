"""Evaluate FLAME on the OxHyperSyntheticCH4 (EMIT) test split.

Evaluation protocol — aligned to HyperspectralViTs (arXiv:2410.17248):
    * ``--eval-mode tile`` (default, paper-aligned): each 512x512 test tile is
      cut into 64x64 windows at stride 32, every window is scored
      independently, and per-pixel predictions/labels/scores are pooled into a
      single global set across all windows of all tiles (overlap pixels counted
      once per window). Binary mask = logits >= 0 (== prob >= 0.5). No
      prediction-side morphological filtering.
    * ``--eval-mode stitch``: full-tile overlap-averaged probability map,
      ``--morph`` optionally applies a morphological opening. Not the paper
      protocol; provided for qualitative comparison.
    * AUPRC = threshold-free average precision over valid pixels.
    * Difficulty subsets (all / easy / hard) per the STARCOP ``difficulty``
      flag; easy/hard include every plume-free tile plus the plume tiles of
      that difficulty (identical background load across subsets).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from flame.datasets.emit import EMITDataset, load_difficulty_map
from flame.eval_common import discover_runs, load_run
from flame.metrics import METRIC_KEYS, metrics_from_arrays
from flame.utils import morph_open, tile_origins

SUBSETS = ['all', 'easy', 'hard']


@torch.no_grad()
def tile_level_collect(model, cube, rgb, gt, mag, patch, stride):
    """Paper-aligned: score each 64x64 window independently, no stitching.

    Returns flat per-pixel (pred, labels, scores) over valid pixels of all
    windows. pred = (logits >= 0); scores = sigmoid(logits).
    """
    _, _, H, W = cube.shape
    pr, lb, sc = [], [], []
    for r in tile_origins(H, patch, stride):
        for c in tile_origins(W, patch, stride):
            sub_cube = cube[:, :, r:r + patch, c:c + patch]
            sub_rgb = rgb[:, :, r:r + patch, c:c + patch]
            if mag is not None:
                logits, _ = model(sub_cube, sub_rgb, mag=mag[:, r:r + patch, c:c + patch])
            else:
                logits, _ = model(sub_cube, sub_rgb)
            logits = logits.squeeze(0).squeeze(0)
            valid = sub_cube.abs().sum(dim=1).squeeze(0) > 0
            if not bool(valid.any()):
                continue
            gtm = gt[r:r + patch, c:c + patch] > 0.5
            pr.append(((logits >= 0)[valid]).cpu().numpy().astype(np.bool_))
            lb.append((gtm[valid]).cpu().numpy().astype(np.bool_))
            sc.append((torch.sigmoid(logits)[valid]).cpu().numpy().astype(np.float16))
    return _cat(pr, np.bool_), _cat(lb, np.bool_), _cat(sc, np.float16)


@torch.no_grad()
def stitch_level_collect(model, cube, rgb, gt, mag, patch, stride, morph, threshold):
    """Full-tile overlap-averaged probability map, optional morph-open."""
    _, _, H, W = cube.shape
    device = cube.device
    acc = torch.zeros(1, 1, H, W, device=device)
    cnt = torch.zeros(1, 1, H, W, device=device)
    for r in tile_origins(H, patch, stride):
        for c in tile_origins(W, patch, stride):
            sub_cube = cube[:, :, r:r + patch, c:c + patch]
            sub_rgb = rgb[:, :, r:r + patch, c:c + patch]
            if mag is not None:
                logits, _ = model(sub_cube, sub_rgb, mag=mag[:, r:r + patch, c:c + patch])
            else:
                logits, _ = model(sub_cube, sub_rgb)
            acc[:, :, r:r + patch, c:c + patch] += torch.sigmoid(logits)
            cnt[:, :, r:r + patch, c:c + patch] += 1.0
    prob = (acc / cnt.clamp(min=1)).squeeze(0).squeeze(0).cpu().numpy()
    valid = (cube.abs().sum(dim=1).squeeze(0) > 0).cpu().numpy()
    pm = prob > threshold
    if morph:
        pm = morph_open(pm)
    gm = gt.cpu().numpy() > 0.5
    return (pm[valid].astype(np.bool_), gm[valid].astype(np.bool_),
            prob[valid].astype(np.float16))


def _cat(chunks, dtype):
    return np.concatenate(chunks) if chunks else np.empty(0, dtype)


def evaluate_run(run: Dict, ds: EMITDataset, diff_map: Dict[str, str], device: str,
                 eval_mode: str, morph: bool, threshold: float,
                 patch: int, stride: int) -> Dict:
    model, _, use_mag = load_run(run, device)
    buf = {s: {'pr': [], 'lb': [], 'sc': []} for s in SUBSETS}

    t0 = time.time()
    for i in range(len(ds)):
        diff = diff_map.get(ds.event_ids[i], 'none')
        cube, rgb, gt, mag = ds[i]
        cube = cube.unsqueeze(0).to(device).float()
        rgb = rgb.unsqueeze(0).to(device).float()
        gt = gt.to(device).float()
        mag_b = mag.unsqueeze(0).to(device).float() if use_mag else None

        if eval_mode == 'tile':
            pr, lb, sc = tile_level_collect(model, cube, rgb, gt, mag_b, patch, stride)
        else:
            pr, lb, sc = stitch_level_collect(
                model, cube, rgb, gt, mag_b, patch, stride, morph, threshold)

        # 'all' always; 'easy'/'hard' get matching plume tiles + every
        # plume-free tile (identical background FP load across subsets).
        targets = ['all'] + ([diff] if diff in ('easy', 'hard') else ['easy', 'hard'])
        for s in targets:
            buf[s]['pr'].append(pr)
            buf[s]['lb'].append(lb)
            buf[s]['sc'].append(sc)
    elapsed = time.time() - t0

    row = dict(seed=run['seed'], n_tiles=len(ds), elapsed_s=elapsed)
    for s in SUBSETS:
        m = metrics_from_arrays(_cat(buf[s]['pr'], np.bool_),
                                _cat(buf[s]['lb'], np.bool_),
                                _cat(buf[s]['sc'], np.float16))
        for k in METRIC_KEYS:
            row[f'{s}__{k}'] = m[k]
    return row


def aggregate(rows: List[Dict], uid: str) -> Dict:
    out = {'uid': uid, 'N': len(rows)}
    for s in SUBSETS:
        for k in METRIC_KEYS:
            col = f'{s}__{k}'
            vals = np.asarray([r[col] for r in rows], dtype=np.float64)
            out[col] = float(np.nanmean(vals))
            out[f'{col}__std'] = float(np.nanstd(vals))
    return out


def write_outputs(per_seed, aggregated, out_dir, eval_mode, morph):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(aggregated).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(out_dir, 'metrics_per_seed.csv'), index=False)
    md_path = os.path.join(out_dir, 'metrics.md')
    with open(md_path, 'w') as f:
        f.write('# FLAME — OxHyperSyntheticCH4 (EMIT) test metrics\n\n')
        f.write(f'Eval protocol: `{eval_mode}` mode, threshold=0.5 (logits>=0), '
                f'morph_open={"on" if (morph and eval_mode == "stitch") else "off"}, '
                f'AUPRC over valid pixels.\n\n')
        for s in SUBSETS:
            f.write(f'## Subset: {s}\n\n')
            f.write('| uid | N | F1 | Precision | Recall | IoU | AUPRC |\n')
            f.write('|---|---|---|---|---|---|---|\n')
            for row in aggregated:
                def fmt(k):
                    return f'{row[f"{s}__{k}"]:.4f} ± {row[f"{s}__{k}__std"]:.4f}'
                f.write(f'| {row["uid"]} | {row["N"]} | {fmt("f1")} | '
                        f'{fmt("precision")} | {fmt("recall")} | {fmt("iou")} | '
                        f'{fmt("auprc")} |\n')
            f.write('\n')
    print(f'Wrote {os.path.join(out_dir, "metrics.csv")}\nWrote {md_path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='FLAME EMIT evaluation')
    ap.add_argument('--uid', action='append', default=None,
                    help='uid in logs/. Repeatable. Default: flame_emit')
    ap.add_argument('--root', default='datasets/oxhyper_synthetic_ch4')
    ap.add_argument('--eval-mode', choices=['tile', 'stitch'], default='tile',
                    help='tile = paper-aligned 64x64 windows; stitch = full-tile average')
    ap.add_argument('--morph', action='store_true',
                    help='apply morph-open (stitch mode only; off by default to match paper)')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='probability threshold (stitch mode; tile mode uses logits>=0)')
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--out-dir', default='results/emit')
    ap.add_argument('--max-test-tiles', type=int, default=None)
    args = ap.parse_args(argv)

    uids = args.uid or ['flame_emit']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read band config (resources_dir / wv_window) from the first run's frozen
    # config so the test dataset matches the model.
    import yaml as _yaml
    res_dir, wv, npy_cache = None, (2004.0, 2478.0), None
    for uid in uids:
        rs = discover_runs(uid)
        if rs:
            dc = _yaml.load(open(rs[0]['config_path']), Loader=_yaml.FullLoader).get('data', {})
            res_dir = dc.get('resources_dir')
            wv = tuple(dc.get('wv_window', wv))
            npy_cache = dc.get('npy_cache_dir')
            break
    test_ds = EMITDataset(root_dir=args.root, split='test', wv_window=wv,
                          patch_size=args.patch, augment=False,
                          max_tiles=args.max_test_tiles, resources_dir=res_dir,
                          npy_cache_dir=npy_cache)
    diff_map = load_difficulty_map(args.root)
    n_easy = sum(1 for e in test_ds.event_ids if diff_map.get(e) == 'easy')
    n_hard = sum(1 for e in test_ds.event_ids if diff_map.get(e) == 'hard')
    print(f'Test tiles: {len(test_ds)}  n_bands={test_ds.n_bands}  '
          f'(easy={n_easy} hard={n_hard} plume-free={len(test_ds) - n_easy - n_hard})')
    print(f'Eval mode: {args.eval_mode}  morph={args.morph}')

    all_rows: List[Dict] = []
    aggs: List[Dict] = []
    for uid in uids:
        runs = discover_runs(uid)
        if not runs:
            print(f'[skip] {uid}: no seeds with best.pt under logs/{uid}/seed_*')
            continue
        per_seed = []
        for run in runs:
            print(f'[{uid}] seed_{run["seed"]} ...')
            row = evaluate_run(run, test_ds, diff_map, device,
                               eval_mode=args.eval_mode, morph=args.morph,
                               threshold=args.threshold,
                               patch=args.patch, stride=args.stride)
            row['uid'] = uid
            per_seed.append(row)
            all_rows.append(row)
            print(f'  [all]  F1={row["all__f1"]:.4f}  P={row["all__precision"]:.4f}  '
                  f'R={row["all__recall"]:.4f}  IoU={row["all__iou"]:.4f}  '
                  f'AUPRC={row["all__auprc"]:.4f}  ({row["elapsed_s"]:.1f}s)')
            print(f'  [easy] F1={row["easy__f1"]:.4f}  AUPRC={row["easy__auprc"]:.4f}   '
                  f'[hard] F1={row["hard__f1"]:.4f}  AUPRC={row["hard__auprc"]:.4f}')
        aggs.append(aggregate(per_seed, uid))

    if not aggs:
        print('No runs evaluated.')
        return
    write_outputs(all_rows, aggs, args.out_dir, args.eval_mode, args.morph)
    with open(os.path.join(args.out_dir, 'eval_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
