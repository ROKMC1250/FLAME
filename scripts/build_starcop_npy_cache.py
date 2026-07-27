"""Pre-stack STARCOP SWIR bands to per-tile .npy for fast training.

Each STARCOP tile stores its 72 SWIR bands as separate TIFFs; stacking them at
every __getitem__ costs 72 reads. This script stacks the 2122-2488 nm bands
once per tile into ``<out>/<tile_id>.npy`` (fp16 by default; the dataset
promotes to fp32 on load). Point ``data.npy_dir`` of the config at ``--out``.

Usage (from repo root):
    python scripts/build_starcop_npy_cache.py --root datasets/starcop \
        --out datasets/starcop_swir_npy --workers 8
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import tifffile as tiff


def swir_wavelengths(tile_dir, wv_range):
    files = sorted(os.listdir(tile_dir))
    band_files = [f for f in files if f.startswith('TOA_AVIRIS_') and f.endswith('nm.tif')]
    all_wv = [int(f.replace('TOA_AVIRIS_', '').replace('nm.tif', '')) for f in band_files]
    return sorted(w for w in all_wv if wv_range[0] <= w <= wv_range[1])


def _one(args):
    tile_dir, out, wv_range, dtype = args
    tid = os.path.basename(tile_dir)
    dst = os.path.join(out, tid + '.npy')
    if os.path.isfile(dst):
        return 0
    try:
        wvs = swir_wavelengths(tile_dir, wv_range)
        bands = [tiff.imread(os.path.join(tile_dir, f'TOA_AVIRIS_{w}nm.tif'))
                 for w in wvs]
        cube = np.stack(bands, axis=0).astype(dtype)
        np.save(dst, cube)
        return 1
    except Exception as e:
        print(f'FAIL {tid}: {e}')
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/starcop')
    ap.add_argument('--out', required=True)
    ap.add_argument('--wv', nargs=2, type=int, default=[2122, 2488])
    ap.add_argument('--fp32', action='store_true', help='store fp32 instead of fp16')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = np.float32 if args.fp32 else np.float16
    tiles = [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
             if os.path.isdir(os.path.join(args.root, d))
             and os.path.isfile(os.path.join(args.root, d, 'labelbinary.tif'))]
    print(f'{len(tiles)} tiles -> {args.out} ({np.dtype(dtype).name})')
    tasks = [(t, args.out, tuple(args.wv), dtype) for t in tiles]
    with Pool(args.workers) as p:
        for i, _ in enumerate(p.imap_unordered(_one, tasks, chunksize=4), 1):
            if i % 100 == 0:
                print(f'{i}/{len(tiles)}')
    print('done')


if __name__ == '__main__':
    main()
