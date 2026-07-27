"""Segmentation metrics shared by training-time validation and evaluation.

Pixel metrics use a global confusion over all evaluated (valid) pixels.
AUPRC is the threshold-free average precision, used by the EMIT protocol
(aligned to HyperspectralViTs, arXiv:2410.17248).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score

METRIC_KEYS = ['f1', 'precision', 'recall', 'iou', 'auprc']


def prf_iou(tp: int, fp: int, fn: int):
    """Precision / recall / F1 / IoU from a confusion triple."""
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    return f1, p, r, iou


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision over valid pixels; NaN if a class is absent."""
    if labels.size == 0 or labels.sum() == 0 or labels.all():
        return float('nan')
    return float(average_precision_score(labels.astype(np.int8),
                                         scores.astype(np.float32)))


def metrics_from_arrays(pred: np.ndarray, labels: np.ndarray,
                        scores: np.ndarray) -> Dict[str, float]:
    """Full metric dict from flat per-pixel arrays.

    Args:
        pred: bool, the binary prediction actually used (already thresholded /
            post-processed). Confusion counts come from this.
        labels: bool ground truth.
        scores: float probability in [0, 1] for AUPRC.
    """
    pred = pred.astype(bool)
    labels = labels.astype(bool)
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    f1, p, r, iou = prf_iou(tp, fp, fn)
    return dict(f1=f1, precision=p, recall=r, iou=iou,
                auprc=safe_auprc(labels, scores), tp=tp, fp=fp, fn=fn)
