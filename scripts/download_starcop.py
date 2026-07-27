"""Download the STARCOP all-bands dataset and arrange the expected layout.

Source: four Hugging Face dataset repos
(https://huggingface.co/collections/previtus/starcop):
    previtus/STARCOP_allbands_Train1 / _Train2 / _Train3 / _Eval

Final layout under --root (default ``datasets/starcop``):
    <root>/<tile_id>/                    all train tiles + all eval/test tiles
    <root>/STARCOP_allbands_Eval/<tile_id>/   the 342 eval tiles (val split)
    <root>/test.csv, train.csv, ...      split CSVs (shipped with the Eval repo)

Training discovers train tiles at the root and excludes every tile that also
appears under ``STARCOP_allbands_Eval/``; evaluation reads ``test.csv`` and
loads its tiles from the root. Eval tiles are HARD-LINKED into the root (no
extra disk usage; falls back to copying across filesystems).

Size: ~633 GB total (~82 GB with --eval-only). Requires a logged-in Hugging
Face client for gated repos: ``hf auth login``.

Usage (from repo root):
    python scripts/download_starcop.py                 # everything
    python scripts/download_starcop.py --eval-only     # test/eval split only
"""
import argparse
import os
import shutil

HF_REPOS = {
    'train1': 'previtus/STARCOP_allbands_Train1',
    'train2': 'previtus/STARCOP_allbands_Train2',
    'train3': 'previtus/STARCOP_allbands_Train3',
    'eval': 'previtus/STARCOP_allbands_Eval',
}
EVAL_DIR = 'STARCOP_allbands_Eval'


def link_or_copy_tree(src, dst):
    """Hard-link a tile directory into dst (copy fallback across devices)."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(d):
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/starcop')
    ap.add_argument('--eval-only', action='store_true',
                    help='download only the eval/test split (~82 GB)')
    ap.add_argument('--workers', type=int, default=8,
                    help='parallel download workers per repo')
    ap.add_argument('--consolidate-only', action='store_true',
                    help='skip downloading; only re-arrange an existing '
                         '<root>/_hf_download stage into the final layout')
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    stage = os.path.join(args.root, '_hf_download')

    if not args.consolidate_only:
        from huggingface_hub import snapshot_download
        repos = HF_REPOS if not args.eval_only else {'eval': HF_REPOS['eval']}
        for name, repo_id in repos.items():
            local_dir = os.path.join(stage, name)
            print(f'[{name}] downloading {repo_id} -> {local_dir}')
            snapshot_download(repo_id=repo_id, repo_type='dataset',
                              local_dir=local_dir, max_workers=args.workers)

    # --- Consolidate ---
    # Train1-3: move tile dirs to the root.
    for name in ('train1', 'train2', 'train3'):
        sub = os.path.join(stage, name)
        if not os.path.isdir(sub):
            continue
        moved = 0
        for entry in sorted(os.listdir(sub)):
            src = os.path.join(sub, entry)
            dst = os.path.join(args.root, entry)
            if os.path.exists(dst):
                continue
            if os.path.isdir(src) and entry.startswith('ang'):
                shutil.move(src, dst)
                moved += 1
            elif entry.endswith('.csv'):
                shutil.move(src, dst)
        print(f'[{name}] moved {moved} tiles to {args.root}')

    # Eval: keep as STARCOP_allbands_Eval/, hard-link tiles into the root,
    # move the split CSVs to the root.
    eval_sub = os.path.join(stage, 'eval')
    if os.path.isdir(eval_sub):
        eval_dst = os.path.join(args.root, EVAL_DIR)
        os.makedirs(eval_dst, exist_ok=True)
        linked = 0
        for entry in sorted(os.listdir(eval_sub)):
            src = os.path.join(eval_sub, entry)
            if os.path.isdir(src) and entry.startswith('ang'):
                dst = os.path.join(eval_dst, entry)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                link_or_copy_tree(dst, os.path.join(args.root, entry))
                linked += 1
            elif entry.endswith('.csv'):
                dst = os.path.join(args.root, entry)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        print(f'[eval] {linked} tiles -> {eval_dst} (hard-linked into root)')

    print(f'\nDone. Layout under {args.root}:')
    n_root = sum(1 for d in os.listdir(args.root) if d.startswith('ang'))
    print(f'  root tiles: {n_root}')
    if os.path.isdir(os.path.join(args.root, EVAL_DIR)):
        n_eval = len(os.listdir(os.path.join(args.root, EVAL_DIR)))
        print(f'  {EVAL_DIR}/: {n_eval} tiles')
    print('  CSVs:', [f for f in os.listdir(args.root) if f.endswith('.csv')])
    print(f'\nNext steps:\n'
          f'  1. (training only) generate mag1c-sas caches: '
          f'python scripts/generate_mag1c_products.py --root {args.root}\n'
          f'  2. (optional) SWIR npy cache: '
          f'python scripts/build_starcop_npy_cache.py --root {args.root} '
          f'--out datasets/starcop_swir_npy')


if __name__ == '__main__':
    main()
