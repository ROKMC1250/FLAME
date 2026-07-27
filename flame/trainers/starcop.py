"""STARCOP trainer.

Two-phase curriculum: aux-only score pretraining for pretrain_epochs (the seg
forward is kept at a tiny weight for DDP gradient flow), then end-to-end
segmentation with the aux loss decaying over decay_frac of the remaining
epochs. Validation is full-tile F1 at sigmoid > val_threshold with a 3x3
morphological opening over valid pixels.
"""
import os

import torch as T
import torch.nn.functional as F
from tqdm import tqdm

from flame.trainers.base import BaseTrainer, Tracker
from flame.utils import get_logger, morph_open


class STARCOPTrainer(BaseTrainer):

    def __init__(self, model, gpu_id, args, log_enabled=True, is_eval=False):
        self.logger = get_logger(self.__class__.__name__, gpu_id)
        super().__init__(model, gpu_id, args, log_enabled, is_eval)
        self.tracker = Tracker(direction='max')
        self.current_epoch = 0

    def _get_optimizer(self):
        return T.optim.AdamW(self.model.parameters(),
                             lr=self.args['train']['lr'], weight_decay=1e-4)

    def _get_scheduler(self):
        total_steps = self.args['train'].get('total_steps', 10000)
        epochs = self.args['train'].get('epoch', self.args['train'].get('epochs', 80))
        warmup_epochs = self.args['train'].get('warmup_epochs', 0)
        steps_per_epoch = max(1, total_steps // max(epochs, 1))
        warmup_steps = warmup_epochs * steps_per_epoch

        cosine_steps = max(1, total_steps - warmup_steps)
        cosine = T.optim.lr_scheduler.CosineAnnealingLR(self.optim, cosine_steps, eta_min=1e-6)
        if warmup_steps <= 0:
            return cosine

        start_factor = self.args['train'].get('warmup_start_factor', 0.1)
        warmup = T.optim.lr_scheduler.LinearLR(
            self.optim, start_factor=start_factor, end_factor=1.0,
            total_iters=warmup_steps)
        return T.optim.lr_scheduler.SequentialLR(
            self.optim, [warmup, cosine], milestones=[warmup_steps])

    def _get_loss_fn(self):
        return T.nn.Identity()

    def _compute_seg_loss(self, logits, gt_target, valid, pw):
        """Segmentation loss (BCE, focal, or Dice+BCE)."""
        loss_type = self.args['train'].get('seg_loss', 'bce')
        focal_gamma = self.args['train'].get('focal_gamma', 0.0)

        if loss_type == 'dice_bce':
            prob = T.sigmoid(logits)
            # Dice loss (on valid pixels only)
            prob_v = prob * valid
            gt_v = gt_target * valid
            inter = (prob_v * gt_v).sum()
            union = prob_v.sum() + gt_v.sum()
            dice_loss = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
            # BCE component (with pw for class balance)
            bce = F.binary_cross_entropy_with_logits(
                logits, gt_target, reduction='none', pos_weight=pw.expand_as(gt_target))
            bce_loss = (bce * valid).sum() / valid.sum().clamp(min=1)
            return dice_loss + bce_loss

        # Standard BCE (or focal)
        bce = F.binary_cross_entropy_with_logits(
            logits, gt_target, reduction='none', pos_weight=pw.expand_as(gt_target))

        if focal_gamma > 0:
            prob = T.sigmoid(logits)
            pt = prob * gt_target + (1 - prob) * (1 - gt_target)
            focal_weight = (1 - pt) ** focal_gamma
            bce = bce * focal_weight

        return (bce * valid).sum() / valid.sum().clamp(min=1)

    def step(self, *batch_data):
        """Training step on a 4-tuple (cube, rgb, gt, mag_norm)."""
        cube, rgb, gt, sas_score = batch_data
        cube = cube.to(self.gpu_id).float()
        rgb = rgb.to(self.gpu_id)
        gt = gt.to(self.gpu_id)
        sas_score = sas_score.to(self.gpu_id)

        pretrain_epochs = self.args['train'].get('pretrain_epochs', 0)
        in_pretrain = self.current_epoch < pretrain_epochs

        logits, score_norm = self.model(cube, rgb)
        gt_target = (gt > 0.5).float().unsqueeze(1)

        # Valid mask (zero-padded regions excluded)
        valid = (cube.abs().sum(dim=1, keepdim=True) > 0).float()

        # Auxiliary loss: spatially-weighted L1 to focus on plume regions
        sas_target = sas_score.unsqueeze(1)
        aux_diff = (score_norm - sas_target).abs() * valid
        spatial_w = 1.0 + sas_target * 10.0
        aux_loss = (aux_diff * spatial_w).sum() / (valid * spatial_w).sum().clamp(min=1)

        # Class balance weight
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

            # Decay aux after pretrain
            epochs = self.args['train'].get('epoch', 80)
            aux_weight = self.args['train'].get('aux_weight', 1.0)
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

    @T.no_grad()
    def validate(self, dl, epoch):
        """F1 validation on full tiles. Val dataset yields 3-tuples."""
        self.logger.info('Validation Phase')
        self.model.eval()

        if not self.args['train']['no_ddp']:
            dl.sampler.set_epoch(epoch)

        tp = T.tensor(0, dtype=T.long, device=self.gpu_id)
        fp = T.tensor(0, dtype=T.long, device=self.gpu_id)
        fn = T.tensor(0, dtype=T.long, device=self.gpu_id)

        pbar = tqdm(dl, desc='Val', disable=not self.is_main_process)
        for batch_data in pbar:
            if len(batch_data) == 3:
                cube, rgb, gt = batch_data
            else:
                cube, gt = batch_data
                rgb = T.zeros(cube.shape[0], 3, cube.shape[2], cube.shape[3])
            cube = cube.to(self.gpu_id).float()
            rgb = rgb.to(self.gpu_id)
            gt = gt.to(self.gpu_id)

            logits, _ = self.model(cube, rgb)
            prob = T.sigmoid(logits)
            valid = cube.abs().sum(dim=1) > 0  # (B, H, W)
            for i in range(cube.shape[0]):
                val_thr = self.args['train'].get('val_threshold', 0.5)
                vm = valid[i].cpu().numpy()
                pm = morph_open(prob[i, 0].cpu().numpy() > val_thr) & vm
                gm = (gt[i].cpu().numpy() > 0.5) & vm
                tp += int((pm & gm).sum())
                fp += int((pm & ~gm).sum())
                fn += int((~pm & gm).sum())
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
            self.logger.info(f'Val F1={f1:.4f} P={p:.4f} R={r:.4f}')

    # -------------------- Visualization --------------------

    def setup_visualization(self, train_dataset, val_dataset,
                            npy_dir=None, pad_to=512, vis_cfg=None):
        """Call before do_training() to enable epoch-end visualization."""
        from flame.vis import select_fixed_vis_indices

        vcfg = vis_cfg or {}
        vis_seed = vcfg.get('seed', 42)

        self._vis_train_ds = train_dataset
        self._vis_val_ds = val_dataset
        self._vis_npy_dir = npy_dir
        self._vis_pad_to = pad_to

        self._vis_val_idx = select_fixed_vis_indices(
            val_dataset.tile_paths,
            n_plume=vcfg.get('n_val_plume', 15),
            n_noplume=vcfg.get('n_val_noplume', 5),
            seed=vis_seed,
        )
        n_train = vcfg.get('n_train', 10)
        self._vis_train_idx = select_fixed_vis_indices(
            train_dataset.tile_paths,
            n_plume=n_train // 2 + n_train % 2,
            n_noplume=n_train // 2,
            seed=vis_seed,
        )
        if self.is_main_process:
            self.logger.info(f'Vis: {len(self._vis_val_idx)} val + '
                             f'{len(self._vis_train_idx)} train tiles')

    def _run_visualization(self, epoch):
        """6-panel with score_norm as panel 2."""
        if not self.can_log or getattr(self, '_vis_val_ds', None) is None:
            return
        from flame.datasets.starcop import load_tile_for_inference
        from flame.vis import save_6panel_visualization

        vis_base = os.path.join(self.log_dir, 'visualizations', f'epoch_{epoch:03d}')
        raw_model = self._raw_model()
        raw_model.eval()

        for label, dataset, indices in [
            ('val', self._vis_val_ds, self._vis_val_idx),
            ('train', self._vis_train_ds, self._vis_train_idx),
        ]:
            if not indices:
                continue
            vis_dir = os.path.join(vis_base, label)
            os.makedirs(vis_dir, exist_ok=True)
            tile_paths = dataset.tile_paths

            for idx in indices:
                tile_path = tile_paths[idx]
                tile_id = os.path.basename(tile_path)
                cube, rgb, gt, _ = load_tile_for_inference(
                    tile_path, npy_dir=self._vis_npy_dir,
                    load_rgb=True, pad_to=self._vis_pad_to)

                with T.no_grad():
                    logits, score_norm = raw_model(
                        cube.unsqueeze(0).to(self.gpu_id),
                        rgb.unsqueeze(0).to(self.gpu_id))
                    prob = T.sigmoid(logits)[0, 0].cpu().numpy()
                    score_np = score_norm[0, 0].cpu().numpy()

                rgb_np = rgb.numpy().transpose(1, 2, 0) if rgb is not None else None
                gt_np = gt.numpy()

                save_6panel_visualization(
                    tile_path, tile_id, score_np, 'Score (norm)',
                    prob, gt_np, vis_dir, epoch,
                    f'FLAME ({label})', rgb_np,
                )
