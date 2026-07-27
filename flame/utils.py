"""Shared utilities: logging, seeding, DDP helpers, morphology, window geometry."""
import logging
import random
import socket

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion

_K = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s::%(levelname)s::%(name)s:%(message)s',
        datefmt='%H:%M:%S',
    )


def get_logger(name, is_disabled=False):
    """Rank-aware logger: pass ``rank != 0`` as is_disabled to silence."""
    class NoOp:
        def __getattr__(self, *args):
            def no_op(*args, **kwargs):
                pass
            return no_op

    if not is_disabled:
        return logging.getLogger(name)
    return NoOp()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def morph_open(m):
    """Binary opening with a 3x3 cross kernel."""
    if not m.any():
        return m
    return binary_dilation(binary_erosion(m, structure=_K), structure=_K)


def tile_origins(size, patch, stride):
    """Sliding-window top-left origins covering [0, size) with a final flush window."""
    origins = list(range(0, max(size - patch, 0) + 1, stride))
    if origins and origins[-1] != size - patch and size > patch:
        origins.append(size - patch)
    if not origins:
        origins = [0]
    return origins
