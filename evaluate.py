"""FLAME evaluation — unified entry point.

Discovers ``logs/<uid>/seed_*/weights/best.pt`` and applies the evaluation
protocol matching the run's dataset:
    * ``starcop`` — full-tile inference, sigmoid > 0.5 + 3x3 morphological
      opening, pixel F1/IoU + emission-rate subsets + tile-level FPR.
    * ``emit`` — paper-aligned tile-level protocol (64/32 windows, logits >= 0,
      no morphological filtering, AUPRC, easy/hard subsets).

The dataset is read from the run's frozen ``config.yaml`` (or forced with
``--dataset``). Remaining flags are forwarded to the protocol evaluator — see
``python evaluate.py --dataset starcop --help`` and ``--dataset emit --help``.

Usage:
    python evaluate.py --uid flame_starcop
    python evaluate.py --uid flame_emit
"""
import argparse
import sys

import yaml

from flame.eval_common import discover_runs


def _infer_dataset(uids):
    for uid in uids:
        for run in discover_runs(uid):
            cfg = yaml.load(open(run['config_path']), Loader=yaml.FullLoader)
            ds = cfg.get('dataset')
            if ds in ('starcop', 'emit'):
                return ds
    return None


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--dataset', choices=['starcop', 'emit'], default=None)
    ap.add_argument('--uid', action='append', default=None)
    known, rest = ap.parse_known_args()

    dataset = known.dataset
    if dataset is None and known.uid:
        dataset = _infer_dataset(known.uid)
    if dataset is None:
        if '-h' in rest or '--help' in rest:
            print(__doc__)
            return
        raise SystemExit(
            'Could not determine the protocol. Pass --dataset starcop|emit, or '
            '--uid <uid> whose logs/<uid>/seed_*/config.yaml has a dataset key.')

    argv = rest
    for uid in (known.uid or []):
        argv = ['--uid', uid] + argv

    if dataset == 'starcop':
        from flame.eval_starcop import main as protocol_main
    else:
        from flame.eval_emit import main as protocol_main
    protocol_main(argv)


if __name__ == '__main__':
    main()
