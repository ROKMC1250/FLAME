"""FLAME qualitative visualization.

Renders per-tile panel figures (RGB, score/mag1c, probability, prediction,
GT, error map) into logs/<uid>/seed_<N>/visualizations/. STARCOP runs
full-tile inference; EMIT stitches 64/32 sliding windows for display only,
reported metrics come from evaluate.py.

Usage:
    python visualize.py --uid flame_starcop --max-tiles 20
    python visualize.py --uid flame_emit --all-tiles --max-tiles 40
"""
import argparse
import os

import pandas as pd
import torch
import yaml

from flame.eval_common import discover_runs, load_run


def parse_args():
    ap = argparse.ArgumentParser(description='FLAME visualization')
    ap.add_argument('--uid', required=True)
    ap.add_argument('--dataset', choices=['starcop', 'emit'], default=None,
                    help='default: read from the run frozen config')
    ap.add_argument('--seed', type=int, default=None,
                    help='specific seed; default = all discovered seeds')
    ap.add_argument('--root', default=None,
                    help='dataset root (default: datasets/starcop or '
                         'datasets/oxhyper_synthetic_ch4)')
    ap.add_argument('--tile-ids', nargs='+', default=None)
    ap.add_argument('--all-tiles', action='store_true',
                    help='visualize every test tile (default: plume tiles only)')
    ap.add_argument('--max-tiles', type=int, default=None)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--morph', action='store_true')
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--ckpt', default='best',
                    help='checkpoint stem to load: best | last | ep009 ...')
    return ap.parse_args()


def _vis_dir(args, run):
    if args.out_dir:
        return os.path.join(args.out_dir, args.uid, f'seed_{run["seed"]}')
    return os.path.join('logs', args.uid, f'seed_{run["seed"]}', 'visualizations')


def run_starcop(args, runs, device):
    import numpy as np
    from flame.datasets.starcop import load_tile_for_inference
    from flame.vis import save_6panel_visualization

    root = args.root or 'datasets/starcop'
    df = pd.read_csv(os.path.join(root, 'test.csv'))
    rows = [(str(r['id']), bool(r.get('has_plume', False))) for _, r in df.iterrows()
            if os.path.isdir(os.path.join(root, str(r['id'])))]
    if args.tile_ids:
        wanted = set(args.tile_ids)
        rows = [r for r in rows if r[0] in wanted]
    elif not args.all_tiles:
        rows = [r for r in rows if r[1]]
    if args.max_tiles is not None:
        rows = rows[:args.max_tiles]
    print(f'Visualizing {len(rows)} tiles for uid={args.uid}')

    for run in runs:
        model, cfg, _ = load_run(run, device)
        npy_dir = cfg.get('data', {}).get('npy_dir')
        vis_dir = _vis_dir(args, run)
        os.makedirs(vis_dir, exist_ok=True)
        mname = f'{args.uid} (seed {run["seed"]}, {args.ckpt})'
        for tid, _ in rows:
            tile_path = os.path.join(root, tid)
            cube, rgb, gt, (H, W) = load_tile_for_inference(
                tile_path, npy_dir=npy_dir, load_rgb=True,
                pad_to=cfg.get('data', {}).get('tile_size', 512))
            with torch.no_grad():
                logits, score = model(cube.unsqueeze(0).to(device).float(),
                                      rgb.unsqueeze(0).to(device).float())
                prob = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()
                score_np = score[0, 0, :H, :W].cpu().numpy()
            rgb_np = rgb[:, :H, :W].numpy().transpose(1, 2, 0)
            save_6panel_visualization(
                tile_path, tid, score_np, 'Score (norm)', prob,
                gt[:H, :W].numpy(), vis_dir, 0, mname, rgb_np)
        print(f'[seed_{run["seed"]}] wrote {len(rows)} panels -> {vis_dir}')


def run_emit(args, runs, device):
    from flame.datasets.emit import EMITDataset, load_difficulty_map
    from flame.vis import full_tile_prob, save_emit_panel

    root = args.root or 'datasets/oxhyper_synthetic_ch4'
    ds = EMITDataset(root_dir=root, split='test',
                     patch_size=args.patch, augment=False)
    diff_map = load_difficulty_map(root)

    if args.tile_ids:
        wanted = set(args.tile_ids)
        idxs = [i for i, e in enumerate(ds.event_ids) if e in wanted]
    elif args.all_tiles:
        idxs = list(range(len(ds)))
    else:  # plume tiles only (easy + hard)
        idxs = [i for i, e in enumerate(ds.event_ids)
                if diff_map.get(e, 'none') in ('easy', 'hard')]
    if args.max_tiles is not None:
        idxs = idxs[:args.max_tiles]
    print(f'Visualizing {len(idxs)} tiles for uid={args.uid}')

    for run in runs:
        model, _, use_mag = load_run(run, device)
        vis_dir = _vis_dir(args, run)
        mname = f'{args.uid} (seed {run["seed"]}, {args.ckpt})'
        for i in idxs:
            eid = ds.event_ids[i]
            cube, rgb, gt, mag = ds[i]
            cube_b = cube.unsqueeze(0).to(device).float()
            rgb_b = rgb.unsqueeze(0).to(device).float()
            mag_b = mag.unsqueeze(0).to(device).float() if use_mag else None
            prob, score = full_tile_prob(model, cube_b, rgb_b, mag_b,
                                         args.patch, args.stride)
            save_emit_panel(eid, diff_map.get(eid, 'none'),
                            rgb.numpy(), mag.numpy(), score, prob, gt.numpy(),
                            vis_dir, mname, args.threshold, args.morph)
        print(f'[seed_{run["seed"]}] wrote {len(idxs)} panels -> {vis_dir}')


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    runs = discover_runs(args.uid)
    if args.seed is not None:
        runs = [r for r in runs if r['seed'] == args.seed]
    if not runs:
        raise SystemExit(f'No runs found for uid={args.uid} (seed={args.seed})')
    if args.ckpt != 'best':
        for run in runs:
            run['ckpt'] = run['ckpt'].replace('best.pt', f'{args.ckpt}.pt')

    dataset = args.dataset
    if dataset is None:
        cfg = yaml.load(open(runs[0]['config_path']), Loader=yaml.FullLoader)
        dataset = cfg.get('dataset')
    if dataset not in ('starcop', 'emit'):
        raise SystemExit('Pass --dataset starcop|emit (frozen config has no dataset key).')

    if dataset == 'starcop':
        run_starcop(args, runs, device)
    else:
        run_emit(args, runs, device)


if __name__ == '__main__':
    main()
