"""STARCOP training pipeline (invoked by train.py for ``dataset: starcop``)."""
import os

import torch as T
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from flame.datasets.starcop import STARCOPDataset, STARCOPTrainDataset
from flame.model import build_model
from flame.trainers.starcop import STARCOPTrainer
from flame.utils import free_port, get_logger, setup_logging


def _ddp_setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    T.cuda.set_device(rank)
    T.cuda.empty_cache()
    init_process_group('nccl', rank=rank, world_size=world_size)


def run(rank, world_size, args, port):
    setup_logging()
    logger = get_logger(__name__, rank)

    if not args['train']['no_ddp']:
        _ddp_setup(rank, world_size, port)

    tcfg, dcfg, mcfg = args['train'], args['data'], args['model']
    vcfg = args.get('vis', {})
    score_divisor = mcfg.get('score_divisor', 1750.0)

    # --- Datasets ---
    dcfg_train = {k: v for k, v in dcfg.items()}
    do_augment = dcfg_train.pop('augment', True)
    dcfg_train['augment'] = False  # augmentation handled by the train wrapper
    cache_small = dcfg_train.pop('cache_small_files', False)
    logger.info(f'Preparing STARCOP dataset (cache_small_files={cache_small})')
    base_tds = STARCOPDataset(**dcfg_train, validation=False, cache_small_files=cache_small)
    tds = STARCOPTrainDataset(base_tds, score_divisor=score_divisor, augment=do_augment)
    logger.info(f'Train: {len(tds)} tiles')

    # Validation: plain STARCOPDataset (no mag1c cache needed)
    vds = STARCOPDataset(**dcfg, validation=True)
    logger.info(f'Val: {len(vds)} tiles')

    # --- DataLoaders ---
    batch_size = tcfg['batch_size']
    n_workers = tcfg['n_workers']
    prefetch = tcfg.get('prefetch_factor', 4)

    if not args['train']['no_ddp']:
        train_sampler = DistributedSampler(tds, shuffle=True)
        tdl = DataLoader(tds, batch_size=batch_size // world_size,
                         sampler=train_sampler, num_workers=n_workers,
                         pin_memory=True, persistent_workers=n_workers > 0,
                         prefetch_factor=prefetch if n_workers > 0 else None)
        val_sampler = DistributedSampler(vds, shuffle=False)
        vdl = DataLoader(vds, batch_size=1, sampler=val_sampler,
                         num_workers=2, pin_memory=True)
    else:
        tdl = DataLoader(tds, batch_size=batch_size, shuffle=True,
                         num_workers=n_workers, pin_memory=True,
                         persistent_workers=n_workers > 0,
                         prefetch_factor=prefetch if n_workers > 0 else None)
        vdl = DataLoader(vds, batch_size=1, shuffle=False,
                         num_workers=2, pin_memory=True)

    # --- Total steps for the cosine schedule ---
    epochs = tcfg.get('epochs', tcfg.get('epoch', 80))
    args['train']['total_steps'] = epochs * len(tdl)

    # --- Model + Trainer ---
    model = build_model(mcfg)
    logger.info(f'FLAME params: {sum(p.numel() for p in model.parameters()):,}')

    trainer = STARCOPTrainer(model, rank, args, log_enabled=not tcfg.get('no_save', False))

    trainer.setup_visualization(
        base_tds, vds,
        npy_dir=dcfg.get('npy_dir'),
        pad_to=dcfg.get('tile_size', 512),
        vis_cfg=vcfg,
    )

    if tcfg.get('model_path'):
        trainer.load_checkpoint(tcfg['model_path'])

    trainer.do_training(tdl, vdl)

    if not args['train']['no_ddp']:
        destroy_process_group()


def launch(cfg, cli):
    """Entry point called by train.py after config/CLI merging."""
    if cli.smoke:
        cfg['data']['val_max_tiles'] = 8
        cfg['train']['epochs'] = 2
        cfg['train']['epoch'] = 2

    if cli.no_ddp:
        run(0, 1, cfg, 0)
        return

    world_size = T.cuda.device_count()
    batch_size = cfg['train']['batch_size']
    assert batch_size % world_size == 0, \
        f'batch_size ({batch_size}) must be divisible by n_gpus ({world_size})'
    port = cli.port or free_port()
    mp.spawn(run, nprocs=world_size, args=(world_size, cfg, port))
