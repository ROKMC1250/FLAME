"""Visualization helpers: panel figures and sliding-window probability maps."""
import os
import random as _random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch

from flame.utils import morph_open, tile_origins


def select_fixed_vis_indices(tile_paths, n_plume=15, n_noplume=5, seed=42):
    """Select deterministic visualization indices stratified by plume/no-plume."""
    plume_idx, noplume_idx = [], []
    for i, tp in enumerate(tile_paths):
        gt_path = os.path.join(tp, 'labelbinary.tif')
        if not os.path.exists(gt_path):
            continue
        gt = tiff.imread(gt_path)
        if np.any(gt > 0):
            plume_idx.append(i)
        else:
            noplume_idx.append(i)

    rng = _random.Random(seed)
    rng.shuffle(plume_idx)
    rng.shuffle(noplume_idx)

    selected = plume_idx[:n_plume] + noplume_idx[:n_noplume]
    return sorted(selected)


def _stretch_rgb_hwc(rgb):
    """Percentile-stretch an (H, W, 3) array to [0, 1] per channel."""
    out = rgb.astype(np.float32).copy()
    for ch in range(3):
        lo, hi = np.percentile(out[..., ch], [2, 98])
        out[..., ch] = np.clip((out[..., ch] - lo) / (hi - lo + 1e-9), 0, 1)
    return out


# ============================================================
# STARCOP 6-panel: RGB | Score | Prob | Masked | GT | TP/FP/FN
# ============================================================

def save_6panel_visualization(tile_path, tile_id, panel2_data, panel2_title,
                              pred_prob_np, gt_np, vis_dir, epoch, model_name,
                              rgb_raw_np=None):
    if rgb_raw_np is not None:
        rgb = _stretch_rgb_hwc(rgb_raw_np)
    else:
        bands = [tiff.imread(os.path.join(tile_path, f'TOA_AVIRIS_{wv}nm.tif')).astype(np.float32)
                 for wv in (640, 550, 460)]
        rgb = _stretch_rgb_hwc(np.stack(bands, axis=-1))

    pred_mask = morph_open(pred_prob_np > 0.5)
    gt_mask = gt_np > 0.5

    # Error map: green=TP, red=FP, blue=FN
    em = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    em[pred_mask & gt_mask] = [0, 1, 0]
    em[pred_mask & ~gt_mask] = [1, 0, 0]
    em[~pred_mask & gt_mask] = [0, 0, 1]

    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    ax[0].imshow(rgb); ax[0].set_title('RGB'); ax[0].axis('off')

    im = ax[1].imshow(panel2_data, cmap='hot')
    ax[1].set_title(f'{panel2_title}\n[{panel2_data.min():.2f}, {panel2_data.max():.2f}]')
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04); ax[1].axis('off')

    ax[2].imshow(pred_prob_np, cmap='hot', vmin=0, vmax=1)
    ax[2].set_title(f'{model_name} prob'); ax[2].axis('off')

    ax[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    ax[3].set_title('Model Masked'); ax[3].axis('off')

    ax[4].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    ax[4].set_title('GT Mask'); ax[4].axis('off')

    ax[5].imshow(em); ax[5].set_title('TP/FP/FN'); ax[5].axis('off')

    fig.suptitle(f'{model_name} (Epoch {epoch + 1}) - {tile_id}', fontsize=10)
    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'{tile_id}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# EMIT 7-panel: RGB | mag1c | score | prob | pred | GT | TP/FP/FN
# ============================================================

@torch.no_grad()
def full_tile_prob(model, cube, rgb, mag, patch, stride):
    """Overlap-averaged full-tile probability map and physics score map."""
    _, _, H, W = cube.shape
    device = cube.device
    acc = torch.zeros(1, 1, H, W, device=device)
    sacc = torch.zeros(1, 1, H, W, device=device)
    cnt = torch.zeros(1, 1, H, W, device=device)
    for r in tile_origins(H, patch, stride):
        for c in tile_origins(W, patch, stride):
            sc = cube[:, :, r:r + patch, c:c + patch]
            sr = rgb[:, :, r:r + patch, c:c + patch]
            if mag is not None:
                lg, score = model(sc, sr, mag=mag[:, r:r + patch, c:c + patch])
            else:
                lg, score = model(sc, sr)
            acc[:, :, r:r + patch, c:c + patch] += torch.sigmoid(lg)
            sacc[:, :, r:r + patch, c:c + patch] += score
            cnt[:, :, r:r + patch, c:c + patch] += 1.0
    cnt = cnt.clamp(min=1)
    prob = (acc / cnt).squeeze(0).squeeze(0).float().cpu().numpy()
    score = (sacc / cnt).squeeze(0).squeeze(0).float().cpu().numpy()
    return prob, score


def save_emit_panel(eid, diff, rgb_chw, mag, score, prob, gt, vis_dir,
                    model_name, threshold, morph):
    rgb = _stretch_rgb_hwc(np.transpose(rgb_chw, (1, 2, 0)))
    valid = np.abs(rgb_chw).sum(0) > 0
    pred = prob > threshold
    if morph:
        pred = morph_open(pred)
    pred = pred & valid
    gtm = (gt > 0.5) & valid

    em = np.zeros((*pred.shape, 3), dtype=np.float32)
    em[pred & gtm] = [0, 1, 0]       # TP green
    em[pred & ~gtm] = [1, 0, 0]      # FP red
    em[~pred & gtm] = [0, 0, 1]      # FN blue

    fig, ax = plt.subplots(1, 7, figsize=(35, 5))
    ax[0].imshow(rgb); ax[0].set_title('RGB'); ax[0].axis('off')

    # mag1c and the physics score head are both normalised to roughly [0, 2];
    # shown side by side so their agreement is directly comparable.
    mvmax = float(np.percentile(mag[mag > 0], 99)) if np.any(mag > 0) else 1.0
    im = ax[1].imshow(mag, cmap='hot', vmin=0, vmax=max(mvmax, 1e-6))
    ax[1].set_title('mag1c (magic30)'); ax[1].axis('off')
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(score, cmap='hot', vmin=0, vmax=2)
    ax[2].set_title('score head (physics)'); ax[2].axis('off')
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    ax[3].imshow(prob, cmap='hot', vmin=0, vmax=1)
    ax[3].set_title(f'{model_name} prob'); ax[3].axis('off')

    ax[4].imshow(pred, cmap='gray', vmin=0, vmax=1)
    ax[4].set_title(f'pred (>{threshold}{" +morph" if morph else ""})'); ax[4].axis('off')

    ax[5].imshow(gtm, cmap='gray', vmin=0, vmax=1)
    ax[5].set_title('GT'); ax[5].axis('off')

    ax[6].imshow(em); ax[6].set_title('TP/FP/FN'); ax[6].axis('off')

    fig.suptitle(f'{model_name} - {eid}  [difficulty={diff}]', fontsize=11)
    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    out = os.path.join(vis_dir, f'{eid}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out
