"""EMIT grid-window training pipeline (invoked by train.py for ``dataset: emit``)."""
import os

import torch as T
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from flame.datasets.emit import (DistributedWeightedSampler, EMITGridDataset,
                                 build_ram_store)
from flame.model import build_model
from flame.trainers.emit import EMITGridTrainer
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

    T.backends.cudnn.benchmark = True          # fixed 64x64 shapes
    T.backends.cuda.matmul.allow_tf32 = True
    T.backends.cudnn.allow_tf32 = True

    tcfg, dcfg, mcfg = args['train'], args['data'], args['model']

    store_dir = dcfg['store_dir']
    root = dcfg['root_dir']
    tds = EMITGridDataset(store_dir, 'train', root_dir=root,
                          windows_csv=dcfg.get('windows_csv',
                                               'train_filtered_v2_tiled_64_32.csv'),
                          patch_size=int(dcfg.get('patch_size', 64)),
                          augment=bool(dcfg.get('augment', True)),
                          max_windows=dcfg.get('max_train_windows'))
    vds = EMITGridDataset(store_dir, 'val', root_dir=root,
                          max_tiles=dcfg.get('max_val_tiles'))
    logger.info(f'grid train windows: {len(tds):,}   val tiles: {len(vds)}')

    bs = tcfg['batch_size']
    nw = tcfg['n_workers']
    prefetch = tcfg.get('prefetch_factor', 4) if nw > 0 else None
    persistent = nw > 0

    # Workers MUST use the 'spawn' mp context: the rank process already holds a
    # CUDA/NCCL context, and fork()ed workers under it leak anonymous memory.
    # Spawn workers start clean; persistent_workers amortises their startup.
    mp_ctx = 'spawn' if nw > 0 else None
    if not args['train']['no_ddp']:
        sampler = DistributedWeightedSampler(tds.sample_weights,
                                             num_samples=len(tds) // world_size,
                                             seed=tcfg.get('seed', 42), rank=rank)
        tdl = DataLoader(tds, batch_size=bs // world_size, sampler=sampler,
                         num_workers=nw, pin_memory=True, drop_last=True,
                         persistent_workers=persistent, prefetch_factor=prefetch,
                         multiprocessing_context=mp_ctx)
        val_sampler = DistributedSampler(vds, shuffle=False)
        vdl = DataLoader(vds, batch_size=1, sampler=val_sampler,
                         num_workers=0, pin_memory=True)
    else:
        sampler = WeightedRandomSampler(T.as_tensor(tds.sample_weights),
                                        num_samples=len(tds), replacement=True)
        tdl = DataLoader(tds, batch_size=bs, sampler=sampler, num_workers=nw,
                         pin_memory=True, drop_last=True,
                         persistent_workers=persistent, prefetch_factor=prefetch,
                         multiprocessing_context=mp_ctx)
        vdl = DataLoader(vds, batch_size=1, shuffle=False, num_workers=0,
                         pin_memory=True)

    epochs = tcfg.get('epochs', tcfg.get('epoch', 50))
    args['train']['total_steps'] = epochs * len(tdl)
    logger.info(f'steps/epoch (per rank): {len(tdl):,}  total_steps: '
                f'{args["train"]["total_steps"]:,}')

    model = build_model(mcfg)
    logger.info(f'FLAME params: {sum(p.numel() for p in model.parameters()):,}')

    trainer = EMITGridTrainer(model, rank, args,
                              log_enabled=not tcfg.get('no_save', False))
    trainer.setup_grid_visualization(tds, vds)
    if tcfg.get('model_path'):
        trainer.load_checkpoint(tcfg['model_path'])

    trainer.do_training(tdl, vdl)

    if not args['train']['no_ddp']:
        destroy_process_group()


def launch(cfg, cli):
    """Entry point called by train.py after config/CLI merging."""
    if cli.smoke:
        cfg['data']['max_train_windows'] = 3000
        cfg['data']['max_val_tiles'] = 8
        cfg['train']['epochs'] = 2
        cfg['train']['epoch'] = 2

    # Build RAM stores BEFORE spawning ranks (single writer, no races).
    dcfg = cfg['data']
    for split in ('train', 'val'):
        build_ram_store(dcfg['root_dir'], split, dcfg['store_dir'],
                        npy_cache_dir=dcfg.get('npy_cache_dir'),
                        wv_window=tuple(dcfg.get('wv_window', (2004.0, 2478.0))),
                        resources_dir=dcfg.get('resources_dir'))

    if cli.no_ddp:
        run(0, 1, cfg, 0)
        return

    world_size = T.cuda.device_count()
    bs = cfg['train']['batch_size']
    assert bs % world_size == 0, \
        f'batch_size ({bs}) must be divisible by n_gpus ({world_size})'
    port = cli.port or free_port()
    mp.spawn(run, nprocs=world_size, args=(world_size, cfg, port))
