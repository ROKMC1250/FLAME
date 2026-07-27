"""Base trainer: DDP wrapping, checkpointing, TensorBoard, epoch loop."""
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch as T
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flame.utils import get_logger


@dataclass
class Tracker:
    last_loss: float = None
    last_metric: float = None
    epoch: int = 0
    step_counter: int = 0
    val_step_counter: int = 0
    best_epoch: int = None
    best_metric: float = None
    direction: str = 'max'

    def inc_step_counter(self):
        self.step_counter += 1

    def inc_val_step_counter(self):
        self.val_step_counter += 1

    def is_metric_better(self, epoch: int) -> bool:
        def _compare(a, b):
            return a > b if self.direction == 'max' else a < b

        if self.best_metric is None or _compare(self.last_metric, self.best_metric):
            self.best_metric = self.last_metric
            self.best_epoch = epoch
            return True
        return False

    def to_dict(self):
        return {
            'last_loss': self.last_loss,
            'last_metric': self.last_metric,
            'epoch': self.epoch,
            'step_counter': self.step_counter,
            'val_step_counter': self.val_step_counter,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'direction': self.direction,
        }


class BaseTrainer(ABC):
    def __init__(self, model: T.nn.Module, gpu_id: int, args: Dict,
                 log_enabled: bool = True, is_eval: bool = False):
        # Normalize epoch/epochs config key
        if 'epochs' in args['train'] and 'epoch' not in args['train']:
            args['train']['epoch'] = args['train']['epochs']

        if getattr(self, 'logger', None) is None:
            self.logger = get_logger(self.__class__.__name__)
        self.model = model
        self.args = args
        self.gpu_id = gpu_id
        self.log_enabled = log_enabled
        self.is_eval = is_eval

        self.uid = args['train']['uid'] if args['train']['uid'] is not None else int(time.time())
        args['train']['uid'] = self.uid
        self.seed = args['train'].get('seed', 42)
        self.loss_fn = self._get_loss_fn()

        if not is_eval:
            self.optim = self._get_optimizer()
            self.scaler = T.GradScaler('cuda')
            self.scheduler = self._get_scheduler()

        if self.can_log:
            self.log_dir = os.path.join(args['train']['log_dir'], f'{self.uid}', f'seed_{self.seed}')
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)
            self.ckpt_dir = os.path.join(self.log_dir, 'weights')

            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.save_config()

        self.tracker = Tracker()
        self.model = self.model.to(self.gpu_id)
        if not args['train']['no_ddp']:
            find_unused = args['train'].get('find_unused_parameters', False)
            self.model = DDP(self.model, device_ids=[self.gpu_id],
                             find_unused_parameters=find_unused)

    @property
    def is_main_process(self):
        return self.gpu_id == 0

    @property
    def can_log(self):
        return self.log_enabled and self.is_main_process

    def _raw_model(self):
        """Unwrap DDP to get the underlying nn.Module."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _get_optimizer(self) -> T.optim.Optimizer:
        return T.optim.AdamW(self.model.parameters(), lr=self.args['train']['lr'],
                             weight_decay=1e-4)

    @abstractmethod
    def _get_scheduler(self) -> T.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()

    @abstractmethod
    def _get_loss_fn(self) -> T.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def step(self, *batch_data) -> T.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def validate(self, dl: DataLoader, epoch: int):
        raise NotImplementedError()

    def write_summary(self, title: str, value: float, step: int):
        if self.can_log:
            self.summary_writer.add_scalar(title, value, step)

    def save_config(self):
        if not self.is_main_process:
            return

        config = self.args

        self.logger.info('======CONFIGURATIONS======')
        for k in config:
            self.logger.info(f'{str(k).upper()}')
            v = config[k]
            if isinstance(v, dict):
                for ik, iv in v.items():
                    self.logger.info(f'\t{ik.upper()}: {iv}')

        config_path = os.path.join(self.log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        self.logger.info(f'Training config saved to {config_path}')

    def save_checkpoint(self, epoch: int, name: str = '', only_model: bool = True):
        if not self.can_log:
            return

        ckpt = {'model': self._raw_model().state_dict(),
                'epoch': epoch,
                'metric': self.tracker.last_metric}

        if not only_model:
            ckpt['optimizer'] = self.optim.state_dict()
            ckpt['scheduler'] = self.scheduler.state_dict()
        if name != '':
            ckpt_path = os.path.join(self.ckpt_dir, f'{name}.pt')
        else:
            ckpt_path = os.path.join(
                self.ckpt_dir, f'epoch{epoch:02}_metric{self.tracker.last_metric:.4f}.pt')

        T.save(ckpt, ckpt_path)
        self.logger.info(f'Checkpoint saved to {ckpt_path}')

    def load_checkpoint(self, ckpt_path: str):
        assert os.path.exists(ckpt_path)
        checkpoint = T.load(ckpt_path, map_location='cpu', weights_only=False)

        state = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self._raw_model().load_state_dict(state, strict=False)

        if 'optimizer' in checkpoint and not self.is_eval:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and not self.is_eval:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.logger.info(f'Loaded checkpoint {ckpt_path}')

    def _run_visualization(self, epoch):
        """Hook called after validation. Override in subclass."""
        pass

    # ==================== Training ====================

    def train(self, dl: DataLoader, epoch: int):
        """Single-epoch training loop. ``yield`` supports multiple validation
        phases within one epoch."""
        self.logger.info('Training Phase')
        self.model.train()

        if not self.args['train']['no_ddp']:
            dl.sampler.set_epoch(epoch)

        from tqdm import tqdm
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(dl, disable=not self.is_main_process)

        for i, batch_data in enumerate(pbar):
            b_loss = self.step(*batch_data)
            self.optim.zero_grad()
            self.scaler.scale(b_loss).backward()
            self.scaler.unscale_(self.optim)
            grad_clip = self.args['train'].get('grad_clip_norm', 1.0)
            if grad_clip and grad_clip > 0:
                T.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()

            loss_val = b_loss.item()
            if self.tracker.step_counter % 50 == 0:
                for k in range(len(self.optim.param_groups)):
                    self.write_summary(f'LR Scheduler/{k}',
                                       self.optim.param_groups[k]['lr'],
                                       self.tracker.step_counter)
                self.write_summary('Train/Batch Loss', loss_val, self.tracker.step_counter)

            self.tracker.inc_step_counter()
            yield i

            total_loss += loss_val
            n_batches += 1
            pbar.set_postfix({'Loss': f'{total_loss / n_batches:.4f}'})

        # Sync loss across GPUs only at epoch end (not every batch)
        avg_loss = T.tensor(total_loss / max(n_batches, 1), device=self.gpu_id)
        if not self.args['train']['no_ddp']:
            T.distributed.all_reduce(avg_loss)
            avg_loss /= T.distributed.get_world_size()
        self.write_summary('Train/Loss', avg_loss, epoch)
        yield -1

    def do_training(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Full training over all epochs."""
        self.logger.info('Begin Training')
        eval_per_epoch = self.args['train'].get('eval_per_epoch', 1)
        epoch = self.args['train'].get('epoch')
        patience = self.args['train'].get('patience', -1)
        ckpt_interval = self.args['train'].get('ckpt_interval', epoch)
        vis_interval = self.args['train'].get('vis_interval', 1)
        eval_idx = [len(train_dataloader) // eval_per_epoch * i
                    for i in range(1, eval_per_epoch)]

        early_stop = False
        for epoch_idx in range(epoch):
            self.logger.info(f'Epoch {epoch_idx + 1}/{epoch}')

            for step in self.train(train_dataloader, epoch_idx):
                if step in eval_idx or step == -1:
                    self.validate(val_dataloader, epoch_idx)
                    # Always visualize on the final epoch so the last
                    # qualitative snapshot is saved.
                    do_vis = (
                        (epoch_idx + 1) % vis_interval == 0
                        or (epoch_idx + 1) == epoch
                    )
                    if do_vis:
                        self._run_visualization(epoch_idx)

                    # Checkpointing
                    self.save_checkpoint(epoch_idx + 1, 'last')
                    if self.tracker.is_metric_better(epoch_idx + 1):
                        self.save_checkpoint(epoch_idx + 1, 'best')
                        if self.is_main_process:
                            self.logger.info(
                                f'*** Best metric={self.tracker.best_metric:.4f} '
                                f'ep{epoch_idx + 1} ***')
                    else:
                        if patience > 0 and epoch_idx + 1 - self.tracker.best_epoch > patience:
                            early_stop = True
                            break

                    # Periodic checkpoint
                    if ckpt_interval and (epoch_idx + 1) % ckpt_interval == 0:
                        self.save_checkpoint(epoch_idx + 1, f'ep{epoch_idx:03d}')

            if self.is_main_process:
                best_ep = self.tracker.best_epoch or 0
                best_m = self.tracker.best_metric or 0
                no_imp = epoch_idx + 1 - best_ep if best_ep else 0
                self.logger.info(
                    f'Best: ep{best_ep} metric={best_m:.4f} '
                    f'(no_improve={no_imp}/{patience})')

            self.logger.info('Epoch complete\n')

            if early_stop:
                self.logger.info(
                    f'Early stopping: no improvement for {patience} epochs.')
                break

        self.logger.info(
            f'Best result at epoch {self.tracker.best_epoch} '
            f'with metric {self.tracker.best_metric:.4f}')

        if self.can_log:
            with open(os.path.join(self.log_dir, 'result.yaml'), 'w') as f:
                yaml.dump(self.tracker.to_dict(), f)
            self.logger.info(f'Result saved to {self.log_dir}/result.yaml')
