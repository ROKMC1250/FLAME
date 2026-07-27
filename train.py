"""FLAME training entry point.

The dataset key of the config (starcop or emit) selects the training pipeline.

Usage:
    python train.py --config configs/flame_starcop.yaml --seed 42
    python train.py --config configs/flame_emit.yaml    --seed 42
    python train.py --config configs/flame_emit.yaml --uid dbg --no-ddp --no-save --smoke
"""
import argparse
import random

import yaml

from flame.utils import seed_everything


def parse_args():
    pa = argparse.ArgumentParser(description='FLAME training')
    pa.add_argument('--config', type=str, required=True)
    pa.add_argument('--uid', type=str, default=None,
                    help='run id; checkpoints go to logs/<uid>/seed_<seed>/')
    pa.add_argument('--model-path', type=str, default=None,
                    help='checkpoint to resume from')
    pa.add_argument('--no-ddp', action='store_true')
    pa.add_argument('--no-save', action='store_true',
                    help='disable logging/checkpoints')
    pa.add_argument('--smoke', action='store_true',
                    help='tiny debug run (few samples, 2 epochs)')
    pa.add_argument('--port', type=int, default=None)
    pa.add_argument('--seed', type=int, default=None)
    pa.add_argument('--lr', type=float, default=None, help='override train.lr')
    return pa.parse_args()


def main():
    cli = parse_args()
    cfg = yaml.load(open(cli.config, 'r'), Loader=yaml.FullLoader)

    dataset = cfg.get('dataset')
    if dataset not in ('starcop', 'emit'):
        raise SystemExit(
            f"config must set a top-level `dataset: starcop | emit` key "
            f"(got {dataset!r})")

    # CLI overrides
    if cli.uid is not None:
        cfg['train']['uid'] = cli.uid
    if cli.seed is not None:
        cfg['train']['seed'] = cli.seed
    if cli.lr is not None:
        cfg['train']['lr'] = cli.lr
    if cli.model_path is not None:
        cfg['train']['model_path'] = cli.model_path
    cfg['train']['no_ddp'] = cli.no_ddp
    cfg['train']['no_save'] = cli.no_save
    if cfg['train'].get('uid') is None:
        cfg['train']['uid'] = f'flame_{dataset}_{random.randint(0, 99999)}'

    seed_everything(cfg['train'].get('seed', 42))

    if dataset == 'starcop':
        from flame.train_starcop import launch
    else:
        from flame.train_emit import launch
    launch(cfg, cli)


if __name__ == '__main__':
    main()
