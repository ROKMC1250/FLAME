"""Pre-decode EMIT cubes to .npy (fp16) for fast RAM-store builds.

The ENVI cube decode (all 86 bands, ~90 MB per tile) dominates the one-time
RAM-store build. This pre-decodes each tile's band-sliced cube once to
``<out>/<event_id>.npy``. Point ``data.npy_cache_dir`` of the config at
``--out``. Optional - without it the store build decodes ENVI directly.

Caches the band-SLICED cube (per resources/wv window), BEFORE nan/scale - the
dataset applies nan_to_num + CUBE_SCALE on load.

Usage (from repo root):
    python scripts/build_emit_npy_cache.py --root datasets/oxhyper_synthetic_ch4 \
        --out datasets/emit_npy64 --workers 16
"""
import argparse
import glob
import os
import re
from multiprocessing import Pool

import numpy as np


def load_band_indices(resources_dir, wv):
    p = os.path.join(resources_dir, 'band_indices.npy')
    if os.path.isfile(p):
        return np.load(p).astype(np.int64)
    return None  # fall back to wv mask below


def _one(args):
    import spectral as spy
    eid, root, out, band_idx, wv = args
    dst = os.path.join(out, eid + '.npy')
    if os.path.isfile(dst):
        return 0
    hdr = os.path.join(root, eid, 'B.hdr')
    try:
        img = spy.open_image(hdr).load()
        cube = np.asarray(img, dtype=np.float32)            # (H, W, 86)
        if band_idx is None:
            bn = spy.open_image(hdr).metadata.get('band names', [])
            wvs = np.array([float(re.match(r'^([0-9.]+)', s.strip()).group(1)) for s in bn])
            bi = np.where((wvs >= wv[0]) & (wvs <= wv[1]))[0]
        else:
            bi = band_idx
        cube = cube[:, :, bi]                                # (H, W, n)
        cube = np.transpose(cube, (2, 0, 1))                 # (n, H, W)
        np.save(dst, cube.astype(np.float16))
        return 1
    except Exception as e:
        print(f'FAIL {eid}: {e}')
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/oxhyper_synthetic_ch4')
    ap.add_argument('--out', required=True)
    ap.add_argument('--resources-dir', default='resources/emit')
    ap.add_argument('--wv', nargs=2, type=float, default=[2004.0, 2478.0])
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    band_idx = load_band_indices(args.resources_dir, args.wv)
    eids = [d for d in sorted(os.listdir(args.root))
            if os.path.isdir(os.path.join(args.root, d))
            and os.path.isfile(os.path.join(args.root, d, 'B.hdr'))]
    print(f'{len(eids)} tiles -> {args.out}  '
          f'(bands={len(band_idx) if band_idx is not None else "wv"})')
    tasks = [(e, args.root, args.out, band_idx, tuple(args.wv)) for e in eids]
    done = 0
    with Pool(args.workers) as p:
        for _ in p.imap_unordered(_one, tasks, chunksize=4):
            done += 1
            if done % 100 == 0:
                print(f'{done}/{len(eids)}')
    print(f'cached {len(glob.glob(os.path.join(args.out, "*.npy")))} npy files')


if __name__ == '__main__':
    main()
