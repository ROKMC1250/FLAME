"""Download the OxHyperSyntheticCH4 (EMIT) dataset.

Source: https://huggingface.co/datasets/previtus/OxHyperSyntheticCH4
(~1,200 event directories + split CSVs; each event ships the ENVI cube
``B``/``B.hdr``, RGB TIFFs, the precomputed mag1c product
``B_magic30_tile.tif``, and ``labelbinary.tif``).

The Hub layout already matches what the code expects, so this is a plain
snapshot into --root (default ``datasets/oxhyper_synthetic_ch4``) — no
re-arrangement and no mag1c generation needed.

Usage (from repo root):
    python scripts/download_emit.py
    python scripts/download_emit.py --splits-only    # just the 4 CSVs
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/oxhyper_synthetic_ch4')
    ap.add_argument('--splits-only', action='store_true',
                    help='download only the split CSVs (layout preview)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    patterns = ['*.csv'] if args.splits_only else None
    print(f'downloading previtus/OxHyperSyntheticCH4 -> {args.root}')
    snapshot_download(repo_id='previtus/OxHyperSyntheticCH4', repo_type='dataset',
                      local_dir=args.root, max_workers=args.workers,
                      allow_patterns=patterns)

    csvs = [f for f in os.listdir(args.root) if f.endswith('.csv')]
    n_tiles = sum(1 for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d))
                  and os.path.isfile(os.path.join(args.root, d, 'B.hdr')))
    print(f'\nDone. {n_tiles} event dirs, CSVs: {sorted(csvs)}')
    print('Optional speed-up for training: '
          f'python scripts/build_emit_npy_cache.py --root {args.root} '
          f'--out datasets/emit_npy64')


if __name__ == '__main__':
    main()
