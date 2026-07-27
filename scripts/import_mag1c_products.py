"""Import mag1c-sas products as per-tile training caches.

STARCOP training supervises the physics score head with the mag1c matched
filter run with 1% covariance subsampling (``mag1c --sample 0.01``,
"mag1c-sas"). This script converts previously generated product TIFFs
(``<products>/<tile_id>/mag1c_tile_sampled-0.01.tif``, produced with the
methane filters benchmark, https://github.com/zaitra/methane-filters-benchmark)
into the ``<root>/<tile_id>/mag1c_sas_cache.npy`` files that
``STARCOPTrainDataset`` expects.

Usage (from repo root):
    python scripts/import_mag1c_products.py \
        --products <path>/products --root datasets/starcop
"""
import argparse
import os

import numpy as np
import tifffile as tiff

PRODUCT_FILE = 'mag1c_tile_sampled-0.01.tif'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--products', required=True,
                    help='directory with <tile_id>/mag1c_tile_sampled-0.01.tif')
    ap.add_argument('--root', default='datasets/starcop',
                    help='STARCOP dataset root (cache written next to each tile)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    n_done = n_skip = n_miss = 0
    for tid in sorted(os.listdir(args.root)):
        tile_dir = os.path.join(args.root, tid)
        if not os.path.isdir(tile_dir) or \
                not os.path.isfile(os.path.join(tile_dir, 'labelbinary.tif')):
            continue
        dst = os.path.join(tile_dir, 'mag1c_sas_cache.npy')
        if os.path.isfile(dst) and not args.overwrite:
            n_skip += 1
            continue
        src = os.path.join(args.products, tid, PRODUCT_FILE)
        if not os.path.isfile(src):
            n_miss += 1
            continue
        mag = tiff.imread(src).astype(np.float32)
        if mag.ndim == 3:
            mag = mag[0]
        np.save(dst, mag)
        n_done += 1

    print(f'written={n_done} skipped(existing)={n_skip} missing_product={n_miss}')
    if n_miss:
        print('Tiles without a product TIFF cannot be used for training; '
              'generate the missing products first.')


if __name__ == '__main__':
    main()
