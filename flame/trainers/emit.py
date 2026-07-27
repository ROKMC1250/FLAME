"""EMIT grid-window trainer.

Matches the HyperspectralViTs training regime: 64x64/stride-32 grid windows,
about half plume windows per epoch, BCE with capped pos_weight, bf16 autocast
(SpectralConv2d keeps its FFT in fp32). Validation reproduces the test
protocol (window pooling, logits >= 0, no morphological filtering) so model
selection matches the reported metric. With model.use_mag_in_seg the raw
mag1c map is fed to the seg head in place of the physics score.
"""
from __future__ import annotations

import os

import numpy as np
import torch as T
import torch.nn.functional as F
from tqdm import tqdm

from flame.trainers.base import BaseTrainer, Tracker
from flame.utils import get_logger, tile_origins


class EMITGridTrainer(BaseTrainer):

    def __init__(self, model, gpu_id, args, log_enabled=True, is_eval=False):
        self.logger = get_logger(self.__class__.__name__, gpu_id)
        super().__init__(model, gpu_id, args, log_enabled, is_eval)
        self.tracker = Tracker(direction='max')
        self.current_epoch = 0
        self.score_divisor = float(args['model'].get('score_divisor', 1750.0))
        self.val_patch = int(args['data'].get('patch_size', 64))
        self.val_stride = int(args['data'].get('stride', 32))
        amp = str(args['train'].get('amp', 'bf16')).lower()
        self._amp_on = amp in ('bf16', 'bfloat16') and T.cuda.is_available()
        self._amp_dtype = T.bfloat16

    # -------------------- BaseTrainer hooks --------------------

    def _get_optimizer(self):
        return T.optim.AdamW(self.model.parameters(),
                             lr=self.args['train']['lr'], weight_decay=1e-4)

    def _get_scheduler(self):
        total = self.args['train'].get('total_steps', 10000)
        return T.optim.lr_scheduler.CosineAnnealingLR(self.optim, total, eta_min=1e-6)

    def _get_loss_fn(self):
        return T.nn.Identity()

    def _compute_seg_loss(self, logits, gt_target, valid, pw):
        loss_type = self.args['train'].get('seg_loss', 'bce')
        focal_gamma = self.args['train'].get('focal_gamma', 0.0)

        if loss_type == 'dice_bce':
            # dice_weight scales the soft-Dice term relative to BCE.
            dice_w = self.args['train'].get('dice_weight', 1.0)
            prob = T.sigmoid(logits)
            prob_v = prob * valid
            gt_v = gt_target * valid
            inter = (prob_v * gt_v).sum()
            union = prob_v.sum() + gt_v.sum()
            dice_loss = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
            bce = F.binary_cross_entropy_with_logits(
                logits, gt_target, reduction='none', pos_weight=pw.expand_as(gt_target))
            bce_loss = (bce * valid).sum() / valid.sum().clamp(min=1)
            return dice_w * dice_loss + bce_loss

        bce = F.binary_cross_entropy_with_logits(
            logits, gt_target, reduction='none', pos_weight=pw.expand_as(gt_target))
        if focal_gamma > 0:
            prob = T.sigmoid(logits)
            pt = prob * gt_target + (1 - prob) * (1 - gt_target)
            bce = bce * (1 - pt) ** focal_gamma
        return (bce * valid).sum() / valid.sum().clamp(min=1)

    # -------------------- Training step --------------------

    def step(self, *batch_data):
        with T.autocast('cuda', dtype=self._amp_dtype, enabled=self._amp_on):
            return self._step_inner(*batch_data)

    def _step_inner(self, *batch_data):
        cube, rgb, gt, mag = batch_data
        cube = cube.to(self.gpu_id).float()
        rgb = rgb.to(self.gpu_id).float()
        gt = gt.to(self.gpu_id).float()
        mag = mag.to(self.gpu_id).float()

        # mag1c -> normalised auxiliary target for the physics score head.
        mag_norm = T.clamp(mag / self.score_divisor, 0.0, 2.0)

        pretrain_epochs = self.args['train'].get('pretrain_epochs', 0)
        in_pretrain = self.current_epoch < pretrain_epochs

        # mag1c input dropout: zero the mag1c channel for a random fraction of
        # samples so the seg head cannot follow mag1c alone and the SWIR path
        # must learn plume signal itself. mag1c is still always supplied at
        # inference; default 0.0 = off.
        mag_model = mag
        mag_drop = self.args['train'].get('mag_dropout', 0.0)
        if mag_drop > 0:
            keep = (T.rand(mag.shape[0], 1, 1, device=mag.device) >= mag_drop).float()
            mag_model = mag * keep

        if self.args['model'].get('use_mag_in_seg', False):
            logits, score_norm = self.model(cube, rgb, mag=mag_model)
        else:
            logits, score_norm = self.model(cube, rgb)
        gt_target = (gt > 0.5).float().unsqueeze(1)

        valid = (cube.abs().sum(dim=1, keepdim=True) > 0).float()

        # Aux loss only on plume regions: pushing score_norm toward 0 on
        # non-plume pixels combined with a clamp creates a gradient trap that
        # drives the score head to a trivial all-zero solution. Restrict
        # supervision to where the matched filter has signal (mag_norm > 0.05).
        sas_target = mag_norm.unsqueeze(1)
        plume_mask = (sas_target > 0.05).float() * valid
        aux_diff = (score_norm - sas_target).abs() * plume_mask
        aux_loss = aux_diff.sum() / plume_mask.sum().clamp(min=1)
        # Background supervision (aux_bg_weight): push score_norm -> 0 where
        # mag1c has no signal. Without it the score is unsupervised in the
        # background and collapses to a near-uniform value. Weight kept < 1 to
        # avoid the trivial all-zero collapse. Default 0.0 = off.
        aux_bg_w = self.args['train'].get('aux_bg_weight', 0.0)
        if aux_bg_w > 0:
            bg_mask = (sas_target <= 0.05).float() * valid
            aux_bg = (score_norm.abs() * bg_mask).sum() / bg_mask.sum().clamp(min=1)
            aux_loss = aux_loss + aux_bg_w * aux_bg

        pw_max = self.args['train'].get('pw_max', 50)
        n_pos = (gt_target * valid).sum().clamp(min=1)
        n_neg = ((1 - gt_target) * valid).sum().clamp(min=1)
        pw = (n_neg / n_pos).clamp(max=pw_max)

        if in_pretrain:
            seg_loss = self._compute_seg_loss(logits, gt_target, valid, pw)
            loss = aux_loss + 0.01 * seg_loss
            gamma = 1.0
        else:
            seg_loss = self._compute_seg_loss(logits, gt_target, valid, pw)
            epochs = self.args['train'].get('epoch', 50)
            aux_weight = self.args['train'].get('aux_weight', 0.0)
            remaining = epochs - pretrain_epochs
            phase2_ep = self.current_epoch - pretrain_epochs
            decay_frac = self.args['train'].get('decay_frac', 0.5)
            decay_epochs = max(1, int(decay_frac * remaining))
            gamma = max(0.0, 1.0 - phase2_ep / decay_epochs)
            loss = seg_loss + aux_weight * gamma * aux_loss

        if self.tracker.step_counter % 50 == 0:
            self.write_summary('Train/seg_loss', seg_loss.item(), self.tracker.step_counter)
            self.write_summary('Train/aux_loss', aux_loss.item(), self.tracker.step_counter)
            self.write_summary('Train/gamma', gamma, self.tracker.step_counter)

        return loss

    def train(self, dl, epoch):
        self.current_epoch = epoch
        yield from super().train(dl, epoch)

    # -------------------- Validation (test protocol) --------------------

    @T.no_grad()
    def validate(self, dl, epoch):
        """Test-protocol validation: 64/32 window pooling, logits >= 0, no morph."""
        self.logger.info('Validation Phase (test-protocol: 64/32 window pooling, no morph)')
        self.model.eval()

        if not self.args['train']['no_ddp']:
            dl.sampler.set_epoch(epoch)

        ps, st = self.val_patch, self.val_stride
        win_bs = int(self.args['train'].get('val_window_batch', 256))
        use_mag = self.args['model'].get('use_mag_in_seg', False)

        tp = T.tensor(0, dtype=T.long, device=self.gpu_id)
        fp = T.tensor(0, dtype=T.long, device=self.gpu_id)
        fn = T.tensor(0, dtype=T.long, device=self.gpu_id)

        pbar = tqdm(dl, desc='Val', disable=not self.is_main_process)
        for cube, rgb, gt, mag in pbar:
            cube = cube.to(self.gpu_id, non_blocking=True).float()
            rgb = rgb.to(self.gpu_id, non_blocking=True).float()
            gt = gt.to(self.gpu_id, non_blocking=True).float()
            mag = mag.to(self.gpu_id, non_blocking=True).float() if use_mag else None
            B, C, H, W = cube.shape
            assert B == 1, 'val batch_size must be 1 (one full tile per item)'
            origins = [(r, c) for r in tile_origins(H, ps, st)
                       for c in tile_origins(W, ps, st)]
            for i in range(0, len(origins), win_bs):
                chunk = origins[i:i + win_bs]
                sc = T.stack([cube[0, :, r:r + ps, c:c + ps] for r, c in chunk])
                sr = T.stack([rgb[0, :, r:r + ps, c:c + ps] for r, c in chunk])
                gw = T.stack([gt[0, r:r + ps, c:c + ps] for r, c in chunk]) > 0.5
                sm = (T.stack([mag[0, r:r + ps, c:c + ps] for r, c in chunk])
                      if use_mag else None)
                with T.autocast('cuda', dtype=self._amp_dtype, enabled=self._amp_on):
                    if use_mag:
                        logits, _ = self.model(sc, sr, mag=sm)
                    else:
                        logits, _ = self.model(sc, sr)
                valid = sc.abs().sum(dim=1) > 0
                pred = (logits.squeeze(1) >= 0) & valid
                gw = gw & valid
                tp += (pred & gw).sum()
                fp += (pred & ~gw).sum()
                fn += ((~pred) & gw).sum()
            self.tracker.inc_val_step_counter()

        if not self.args['train']['no_ddp']:
            T.distributed.all_reduce(tp)
            T.distributed.all_reduce(fp)
            T.distributed.all_reduce(fn)

        tp_v, fp_v, fn_v = tp.item(), fp.item(), fn.item()
        p = tp_v / max(tp_v + fp_v, 1)
        r = tp_v / max(tp_v + fn_v, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)

        self.tracker.last_metric = f1
        self.tracker.last_loss = 1.0 - f1
        self.write_summary('Validation/F1', f1, epoch)
        self.write_summary('Validation/Precision', p, epoch)
        self.write_summary('Validation/Recall', r, epoch)
        if self.is_main_process:
            self.logger.info(f'Val(test-protocol) F1={f1:.4f} P={p:.4f} R={r:.4f}')

    # -------------------- Visualization --------------------

    def setup_grid_visualization(self, train_ds, val_ds,
                                 n_train_plume=3, n_train_empty=1,
                                 n_val_plume=4, n_val_empty=2, min_px=150):
        def _select(ds, n_plume, n_empty):
            sums = ds.tile_gt_sums()
            plume = [int(i) for i in np.argsort(-sums) if sums[i] >= min_px][:n_plume]
            empty = [int(i) for i in np.where(sums == 0)[0][:n_empty]]
            return plume + empty

        self._vis_specs = []
        if train_ds is not None:
            for i in _select(train_ds, n_train_plume, n_train_empty):
                self._vis_specs.append(('train', train_ds, i))
        if val_ds is not None:
            for i in _select(val_ds, n_val_plume, n_val_empty):
                self._vis_specs.append(('val', val_ds, i))
        if self.is_main_process:
            n_tr = sum(1 for s in self._vis_specs if s[0] == 'train')
            self.logger.info(f'Vis tiles: {n_tr} train + {len(self._vis_specs) - n_tr} val')

    @T.no_grad()
    def _run_visualization(self, epoch):
        if not self.can_log or not getattr(self, '_vis_specs', None):
            return
        from flame.vis import full_tile_prob, save_emit_panel
        model = self._raw_model()
        was_training = model.training
        model.eval()
        use_mag = self.args['model'].get('use_mag_in_seg', False)
        vis_dir = os.path.join(self.log_dir, 'visualizations', f'epoch_{epoch:03d}')
        thr = self.args['train'].get('val_threshold', 0.5)
        for split, ds, idx in self._vis_specs:
            cube, rgb, gt, mag = ds.get_full_tile(idx)
            cube_b = cube.unsqueeze(0).to(self.gpu_id)
            rgb_b = rgb.unsqueeze(0).to(self.gpu_id)
            mag_b = mag.unsqueeze(0).to(self.gpu_id) if use_mag else None
            with T.autocast('cuda', dtype=self._amp_dtype, enabled=self._amp_on):
                prob, score = full_tile_prob(model, cube_b, rgb_b, mag_b,
                                             self.val_patch, self.val_stride)
            eid = ds.event_ids[idx]
            save_emit_panel(f'{split}_{eid}', 'n/a', rgb.numpy(), mag.numpy(),
                            score.astype(np.float32), prob.astype(np.float32), gt.numpy(),
                            vis_dir, f'{self.uid} (seed {self.seed}, ep{epoch + 1})',
                            thr, morph=False)
        if was_training:
            model.train()
        self.logger.info(f'Visualizations -> {vis_dir}')
