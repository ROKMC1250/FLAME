"""Generate per-tile mag1c-sas training caches for STARCOP.

STARCOP training supervises the physics score head with the mag1c matched
filter run with 1% covariance subsampling ("mag1c-sas"). The products are
computed by the methane filters benchmark
(https://github.com/zaitra/methane-filters-benchmark, ``mag1c --sample 0.01``
on the 2122-2488 nm bands), which this script drives end to end:

    1. clone the benchmark into --benchmark-dir (skipped if present);
    2. run its ``benchmark/create_filters_for_starcop.py`` for every tile
       listed in --csv (mag1c products only, no CEM/MF/ACE);
    3. import each ``mag1c_tile_sampled-0.01.tif`` product as
       ``<root>/<tile_id>/mag1c_sas_cache.npy`` (the file
       ``STARCOPTrainDataset`` expects).

Requires a CUDA GPU and the benchmark's extra deps:
    pip install pysptools imagecodecs

Usage (from repo root):
    python scripts/generate_mag1c_products.py --root datasets/starcop --csv train.csv
    python scripts/generate_mag1c_products.py --root datasets/starcop --csv test.csv
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import tifffile as tiff
import yaml

BENCHMARK_URL = 'https://github.com/zaitra/methane-filters-benchmark'
SAS_PRODUCT = 'mag1c_tile_sampled-{sample}.tif'


def ensure_benchmark(benchmark_dir):
    if os.path.isfile(os.path.join(benchmark_dir, 'benchmark',
                                   'create_filters_for_starcop.py')):
        return
    print(f'cloning {BENCHMARK_URL} -> {benchmark_dir}')
    subprocess.run(['git', 'clone', '--depth', '1', BENCHMARK_URL, benchmark_dir],
                   check=True)


def run_benchmark(benchmark_dir, root, csv_path, out_dir, sample, resume):
    config = {
        'COLUMN': False,
        'CREATE_TILE_MAG1C': True,        # required for the sampled variant
        'CREATE_SAMPLED_MAG1C': True,
        'SAMPLE_PERCENT': sample,
        'SELECT_BANDS': False,
        'BANDS_N': 72,
        'STRATEGY': 'highest-transmittance',
        'CREATE_OTHER_FILTERS': False,    # mag1c only — no CEM/MF/ACE
        'RESUME': resume,
        'PRECISION': 64,
        'USE_SPED_UP_VERSIONS_OF_FILTERS': True,
        'wavelengths_range': [2122, 2488],
        'csv_path': os.path.abspath(csv_path),
        'input_data_path': os.path.abspath(root),
        'output_data_path': os.path.abspath(out_dir),
    }
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.abspath(os.path.join(out_dir, 'mag1c_config.yaml'))
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = [sys.executable,
           os.path.join('benchmark', 'create_filters_for_starcop.py'),
           '--config', config_path]
    print('running:', ' '.join(cmd), f'(cwd={benchmark_dir})')
    result = subprocess.run(cmd, cwd=benchmark_dir)
    if result.returncode != 0:
        sys.exit(f'ERROR: benchmark run failed (exit {result.returncode})')


def import_caches(out_dir, root, sample, overwrite=False):
    """Copy <note_dir>/<tile>/mag1c_tile_sampled-<s>.tif -> <root>/<tile>/mag1c_sas_cache.npy."""
    product_name = SAS_PRODUCT.format(sample=sample)
    # The benchmark writes into a settings-derived subdirectory of out_dir.
    product_files = glob.glob(os.path.join(out_dir, '*', '*', product_name))
    n_done = n_skip = 0
    for src in sorted(product_files):
        tid = os.path.basename(os.path.dirname(src))
        tile_dir = os.path.join(root, tid)
        if not os.path.isdir(tile_dir):
            continue
        dst = os.path.join(tile_dir, 'mag1c_sas_cache.npy')
        if os.path.isfile(dst) and not overwrite:
            n_skip += 1
            continue
        mag = tiff.imread(src).astype(np.float32)
        if mag.ndim == 3:
            mag = mag[0]
        np.save(dst, mag)
        n_done += 1
    print(f'imported {n_done} caches into {root} (skipped existing: {n_skip})')
    return n_done + n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/starcop',
                    help='STARCOP dataset root (tile dirs + CSVs)')
    ap.add_argument('--csv', default='train.csv',
                    help='CSV (relative to --root or absolute) listing tile ids')
    ap.add_argument('--benchmark-dir', default='third_party/methane-filters-benchmark',
                    help='methane-filters-benchmark checkout (cloned if missing)')
    ap.add_argument('--out', default='results/mag1c_products',
                    help='product staging directory (safe to delete afterwards)')
    ap.add_argument('--sample', type=float, default=0.01,
                    help='mag1c covariance sampling fraction')
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--overwrite', action='store_true',
                    help='overwrite existing mag1c_sas_cache.npy files')
    ap.add_argument('--import-only', action='store_true',
                    help='skip generation; only import existing products from --out')
    args = ap.parse_args()

    csv_path = args.csv if os.path.isabs(args.csv) or os.path.isfile(args.csv) \
        else os.path.join(args.root, args.csv)
    if not os.path.isfile(csv_path):
        sys.exit(f'CSV not found: {csv_path}')

    if not args.import_only:
        ensure_benchmark(args.benchmark_dir)
        run_benchmark(args.benchmark_dir, args.root, csv_path, args.out,
                      args.sample, resume=not args.no_resume)

    n = import_caches(args.out, args.root, args.sample, overwrite=args.overwrite)
    if n == 0:
        sys.exit('No products found to import — check the benchmark output.')
    print(f'\nDone. Products remain under {args.out} (delete to reclaim space).')


if __name__ == '__main__':
    main()
