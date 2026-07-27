"""Run discovery and checkpoint loading shared by evaluation and visualization."""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Tuple

import torch
import yaml

from flame.model import FLAME, build_model


def discover_runs(uid: str, log_dir: str = 'logs') -> List[Dict]:
    """Return [{seed, ckpt, config_path}, ...] for ``<log_dir>/<uid>/seed_*/weights/best.pt``."""
    runs = []
    for sd in sorted(glob.glob(os.path.join(log_dir, uid, 'seed_*'))):
        m = re.match(r'seed_(\d+)$', os.path.basename(sd))
        if not m:
            continue
        ckpt = os.path.join(sd, 'weights', 'best.pt')
        cfg = os.path.join(sd, 'config.yaml')
        if os.path.isfile(ckpt) and os.path.isfile(cfg):
            runs.append({'seed': int(m.group(1)), 'ckpt': ckpt, 'config_path': cfg})
    return runs


def load_run(run: Dict, device: str) -> Tuple[FLAME, dict, bool]:
    """Build FLAME from a run's frozen config and load its best.pt.

    Returns (model.eval(), cfg, use_mag). ``use_mag`` mirrors the frozen
    config's ``model.use_mag_in_seg`` — when true, mag1c must be passed to
    ``model(...)`` at inference.
    """
    cfg = yaml.load(open(run['config_path']), Loader=yaml.FullLoader)
    mcfg = cfg['model']
    use_mag = bool(mcfg.get('use_mag_in_seg', False))
    model = build_model(mcfg).to(device).eval()
    state = torch.load(run['ckpt'], map_location=device, weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    return model, cfg, use_mag
