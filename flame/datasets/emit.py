"""EMIT (OxHyperSyntheticCH4) datasets.

Tiles live under <root>/<event_id>/ with the ENVI cube B/B.hdr, RGB tifs, the
mag1c product B_magic30_tile.tif and labelbinary.tif. Splits come from the
CSVs at the dataset root.

EMITDataset returns one item per tile (full 512x512 for val/test, a random
crop for train) and is used for evaluation and for building the RAM store.
EMITGridDataset is the training regime: the deterministic 64x64/stride-32
grid windows from train_filtered_v2_tiled_64_32.csv over a flat fp16 memmap
store, with plume windows balanced to about half of each epoch by a
with-replacement weighted sampler.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

# Default columns we read; CSVs have many more but we only need these.
_CSV_COLS = ['event_id', 'has_plume', 'qplume']

STORE_ARRAYS = ('cube', 'rgb', 'mag', 'gt')


def _parse_emit_band_centers(hdr_path: str) -> np.ndarray:
    import spectral as spy
    img = spy.open_image(hdr_path)
    bn = img.metadata.get('band names', [])
    wvs = []
    for s in bn:
        m = re.match(r'^([0-9.]+)', s.strip())
        if not m:
            raise ValueError(f'Cannot parse band name: {s!r}')
        wvs.append(float(m.group(1)))
    return np.asarray(wvs, dtype=np.float64)


def _load_tif(path: str) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1)
    return arr


class EMITDataset(Dataset):
    """OxHyperSyntheticCH4 tile dataset.

    Args:
        root_dir: dataset root containing ``<event_id>/`` subfolders and split CSVs.
        split: one of ``train``, ``val``, ``test``. Selects the matching CSV.
        csv_name: override CSV filename.
        wv_window: (lo, hi) nm - bands inside this window are kept.
        patch_size: spatial size emitted in ``train`` mode. Ignored in val/test.
        augment: apply random flip/90 rotation in ``train`` mode.
        rgb_bands: filenames of single-band RGB TIFFs in (R, G, B) order.
        max_tiles: cap number of tiles (debug/smoke).
        mag_filename: matched-filter TIFF filename (default ``B_magic30_tile.tif``).
        resources_dir: directory with ``band_indices.npy`` / ``band_centers.npy``
            (default ``resources/emit``).
        npy_cache_dir: pre-decoded band-sliced cubes (scripts/build_emit_npy_cache.py).
    """

    SPLIT_TO_CSV = {
        'train': 'train_filtered_v2.csv',
        'val': 'val_filtered_v2.csv',
        'test': 'test_filtered_v2.csv',
    }

    # EMIT cubes are in reflectance ([0.1, 1.6]); the physics score head expects
    # AVIRIS-NG-scale radiance (~10^2). Scale cubes by this factor so
    # residual x spectrum lands in a usable range.
    CUBE_SCALE: float = 100.0

    def __init__(self,
                 root_dir: str,
                 split: str,
                 csv_name: Optional[str] = None,
                 wv_window: Tuple[float, float] = (2004.0, 2478.0),
                 patch_size: int = 64,
                 augment: bool = False,
                 rgb_bands: Sequence[str] = (
                     'B_EMIT_641nm.tif', 'B_EMIT_551nm.tif', 'B_EMIT_462nm.tif'),
                 max_tiles: Optional[int] = None,
                 mag_filename: str = 'B_magic30_tile.tif',
                 resources_dir: Optional[str] = None,
                 npy_cache_dir: Optional[str] = None):
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'split must be train/val/test, got {split!r}')

        self.root_dir = root_dir
        self.split = split
        self.wv_window = wv_window
        self.patch_size = int(patch_size)
        self.augment = augment and split == 'train'
        self.rgb_bands = tuple(rgb_bands)
        self.mag_filename = mag_filename
        self._npy_cache_dir = npy_cache_dir

        csv_name = csv_name or self.SPLIT_TO_CSV[split]
        csv_path = os.path.join(root_dir, csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'Split CSV not found: {csv_path}')
        df = pd.read_csv(csv_path, usecols=_CSV_COLS, low_memory=False)

        # Filter to event_ids whose tile folder actually exists on disk
        # (handles partial downloads gracefully).
        event_ids: List[str] = []
        for eid in df['event_id'].astype(str).tolist():
            tdir = os.path.join(root_dir, eid)
            if os.path.isdir(tdir) and os.path.isfile(os.path.join(tdir, 'B.hdr')):
                event_ids.append(eid)
        missing = len(df) - len(event_ids)
        if missing > 0:
            print(f'[EMITDataset:{split}] {missing}/{len(df)} tiles missing on '
                  f'disk; using {len(event_ids)}.')

        if max_tiles is not None:
            event_ids = event_ids[:max_tiles]
        if not event_ids:
            raise RuntimeError(f'No tiles found for split={split} under {root_dir}')

        self.event_ids = event_ids

        # Band layout is sensor-fixed across the whole dataset (86 EMIT bands).
        # Some tiles lost wavelength metadata during HF re-packaging and report
        # 'Band 1..86' in their hdr - so prefer the shipped resources files.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        res_dir = resources_dir or os.path.join(repo_root, 'resources', 'emit')
        if not os.path.isabs(res_dir):
            res_dir = os.path.join(repo_root, res_dir)
        idx_path = os.path.join(res_dir, 'band_indices.npy')
        ctr_path = os.path.join(res_dir, 'band_centers.npy')
        if os.path.isfile(idx_path) and os.path.isfile(ctr_path):
            self.band_indices: np.ndarray = np.load(idx_path).astype(np.int64)
            self.band_centers: np.ndarray = np.load(ctr_path).astype(np.float64)
        else:
            sample_hdr = self._find_hdr_with_wavelengths(event_ids, root_dir)
            all_wvs = _parse_emit_band_centers(sample_hdr)
            mask = (all_wvs >= wv_window[0]) & (all_wvs <= wv_window[1])
            self.band_indices = np.where(mask)[0]
            self.band_centers = all_wvs[mask]
        self.n_bands: int = int(self.band_indices.size)
        if self.n_bands == 0:
            raise RuntimeError(f'No EMIT bands inside window {wv_window}')

    @staticmethod
    def _find_hdr_with_wavelengths(event_ids, root_dir):
        for eid in event_ids:
            hdr = os.path.join(root_dir, eid, 'B.hdr')
            try:
                wvs = _parse_emit_band_centers(hdr)
                if wvs.max() > 200:           # plausible wavelength range
                    return hdr
            except (ValueError, RuntimeError):
                continue
        raise RuntimeError(
            'No tile under root has parseable wavelength metadata and no '
            'resources/emit/band_indices.npy was found.')

    def __len__(self) -> int:
        return len(self.event_ids)

    def _load_tile(self, eid: str):
        import spectral as spy
        tdir = os.path.join(self.root_dir, eid)
        # Fast path: pre-decoded band-sliced cube .npy - avoids the per-item
        # ENVI decode of the full 86-band cube.
        npy = (os.path.join(self._npy_cache_dir, eid + '.npy')
               if self._npy_cache_dir else None)
        if npy and os.path.isfile(npy):
            cube = np.load(npy).astype(np.float32)        # (n_bands, H, W), already sliced
        else:
            img = spy.open_image(os.path.join(tdir, 'B.hdr')).load()
            cube = np.asarray(img, dtype=np.float32)
            cube = cube[:, :, self.band_indices]          # (H, W, n_bands)
            cube = np.transpose(cube, (2, 0, 1))          # (n_bands, H, W)
        # ~2.5% of EMIT pixels are NaN (sensor edges, masked clouds). Replace
        # with 0 so the valid mask (cube.abs().sum > 0) excludes them.
        np.nan_to_num(cube, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cube *= self.CUBE_SCALE

        rgb = np.stack([_load_tif(os.path.join(tdir, fn)) for fn in self.rgb_bands],
                       axis=0).astype(np.float32)         # (3, H, W)
        np.nan_to_num(rgb, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        mag = _load_tif(os.path.join(tdir, self.mag_filename)).astype(np.float32)
        np.nan_to_num(mag, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        gt = _load_tif(os.path.join(tdir, 'labelbinary.tif')).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)
        return cube, rgb, mag, gt

    # Probability of biasing the random crop toward a plume pixel when the
    # tile contains one (train mode of this class only; the grid regime below
    # does not use it).
    PLUME_CROP_PROB = 0.7

    def _sample_crop_origin(self, gt, H, W, ps):
        if gt.sum() > 0 and random.random() < self.PLUME_CROP_PROB:
            ys, xs = np.where(gt > 0.5)
            i = random.randrange(len(ys))
            cy, cx = int(ys[i]), int(xs[i])
            jitter = ps // 2
            r = cy - ps // 2 + random.randint(-jitter, jitter)
            c = cx - ps // 2 + random.randint(-jitter, jitter)
            r = max(0, min(r, H - ps))
            c = max(0, min(c, W - ps))
            return r, c
        return (random.randint(0, max(0, H - ps)),
                random.randint(0, max(0, W - ps)))

    @staticmethod
    def _augment_np(cube, rgb, gt, mag):
        if random.random() > 0.5:
            cube = np.flip(cube, -1).copy(); rgb = np.flip(rgb, -1).copy()
            gt = np.flip(gt, -1).copy(); mag = np.flip(mag, -1).copy()
        if random.random() > 0.5:
            cube = np.flip(cube, -2).copy(); rgb = np.flip(rgb, -2).copy()
            gt = np.flip(gt, -2).copy(); mag = np.flip(mag, -2).copy()
        k = random.randint(0, 3)
        if k > 0:
            cube = np.rot90(cube, k, (-2, -1)).copy()
            rgb = np.rot90(rgb, k, (-2, -1)).copy()
            gt = np.rot90(gt, k, (-2, -1)).copy()
            mag = np.rot90(mag, k, (-2, -1)).copy()
        return cube, rgb, gt, mag

    def __getitem__(self, idx: int):
        eid = self.event_ids[idx]
        cube, rgb, mag, gt = self._load_tile(eid)

        if self.split == 'train':
            ps = self.patch_size
            _, H, W = cube.shape
            r, c = self._sample_crop_origin(gt, H, W, ps)
            cube = cube[:, r:r + ps, c:c + ps]
            rgb = rgb[:, r:r + ps, c:c + ps]
            mag = mag[r:r + ps, c:c + ps]
            gt = gt[r:r + ps, c:c + ps]
            if self.augment:
                cube, rgb, gt, mag = self._augment_np(cube, rgb, gt, mag)

        return (
            torch.from_numpy(cube),                       # (n_bands, h, w)
            torch.from_numpy(rgb),                        # (3, h, w)
            torch.from_numpy(gt),                         # (h, w)
            torch.from_numpy(mag),                        # (h, w)
        )


# ============================================================
# RAM store + grid-window training dataset
# ============================================================

def _store_paths(store_dir: str, split: str):
    return {a: os.path.join(store_dir, f'{split}_{a}.npy') for a in STORE_ARRAYS}


def store_complete(store_dir: str, split: str) -> bool:
    ev = os.path.join(store_dir, f'{split}_events.json')
    if not os.path.isfile(ev):
        return False
    paths = _store_paths(store_dir, split)
    return all(os.path.isfile(p) for p in paths.values())


def build_ram_store(root_dir: str, split: str, store_dir: str,
                    npy_cache_dir: Optional[str] = None,
                    wv_window=(2004.0, 2478.0),
                    resources_dir: Optional[str] = None) -> None:
    """One-time consolidation of a split into flat fp16 memmap arrays.

    Datasets open the arrays with ``mmap_mode='r'``; the OS page cache keeps
    them RAM-resident and shared across DDP ranks and dataloader workers, so a
    ``__getitem__`` is a small slice copy - no TIFF/ENVI decode. Skips if the
    store is already present.
    """
    if store_complete(store_dir, split):
        print(f'[ram_store:{split}] already built at {store_dir}')
        return
    os.makedirs(store_dir, exist_ok=True)
    ds = EMITDataset(root_dir=root_dir, split=split, wv_window=wv_window,
                     augment=False, resources_dir=resources_dir,
                     npy_cache_dir=npy_cache_dir)
    eids = ds.event_ids
    n, nb = len(eids), ds.n_bands
    paths = _store_paths(store_dir, split)
    cube_mm = np.lib.format.open_memmap(paths['cube'], mode='w+', dtype=np.float16,
                                        shape=(n, nb, 512, 512))
    rgb_mm = np.lib.format.open_memmap(paths['rgb'], mode='w+', dtype=np.float16,
                                       shape=(n, 3, 512, 512))
    mag_mm = np.lib.format.open_memmap(paths['mag'], mode='w+', dtype=np.float16,
                                       shape=(n, 512, 512))
    gt_mm = np.lib.format.open_memmap(paths['gt'], mode='w+', dtype=np.uint8,
                                      shape=(n, 512, 512))
    for i, eid in enumerate(eids):
        tdir = os.path.join(root_dir, eid)
        npy = (os.path.join(npy_cache_dir, eid + '.npy') if npy_cache_dir else None)
        if npy and os.path.isfile(npy):
            cube = np.load(npy)                       # fp16 (nb, 512, 512), unscaled
            cube = np.nan_to_num(cube.astype(np.float32), nan=0.0,
                                 posinf=0.0, neginf=0.0)
        else:
            import spectral as spy
            img = spy.open_image(os.path.join(tdir, 'B.hdr')).load()
            cube = np.asarray(img, dtype=np.float32)[:, :, ds.band_indices]
            cube = np.transpose(cube, (2, 0, 1))
            np.nan_to_num(cube, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cube_mm[i] = cube.astype(np.float16)
        rgb = np.stack([_load_tif(os.path.join(tdir, fn)) for fn in ds.rgb_bands],
                       axis=0).astype(np.float32)
        np.nan_to_num(rgb, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        rgb_mm[i] = rgb.astype(np.float16)
        mag = _load_tif(os.path.join(tdir, ds.mag_filename)).astype(np.float32)
        np.nan_to_num(mag, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        mag_mm[i] = mag.astype(np.float16)
        gt = _load_tif(os.path.join(tdir, 'labelbinary.tif'))
        gt_mm[i] = (np.nan_to_num(gt.astype(np.float32)) > 0.5).astype(np.uint8)
        if (i + 1) % 100 == 0:
            print(f'[ram_store:{split}] {i + 1}/{n}')
    for mm in (cube_mm, rgb_mm, mag_mm, gt_mm):
        mm.flush()
    with open(os.path.join(store_dir, f'{split}_events.json'), 'w') as f:
        json.dump(eids, f)
    print(f'[ram_store:{split}] built {n} tiles -> {store_dir}')


class EMITGridDataset(Dataset):
    """Grid-window dataset over a prebuilt RAM store.

    split='train': one item per grid window (windows_csv), plume-biased ONLY via
        the external weighted sampler (this class just exposes weights).
    split='val'/'test': one item per full 512x512 tile.
    Returns the same 4-tuple as EMITDataset: (cube, rgb, gt, mag).
    """

    CUBE_SCALE = EMITDataset.CUBE_SCALE  # 100.0

    def __init__(self, store_dir: str, split: str, root_dir: str,
                 windows_csv: str = 'train_filtered_v2_tiled_64_32.csv',
                 patch_size: int = 64, augment: bool = False,
                 max_tiles: Optional[int] = None,
                 max_windows: Optional[int] = None):
        self.store_dir = store_dir
        self.split = split
        self.patch_size = int(patch_size)
        self.augment = augment and split == 'train'

        with open(os.path.join(store_dir, f'{split}_events.json')) as f:
            self.event_ids = json.load(f)
        self._paths = _store_paths(store_dir, split)
        # mmaps are opened lazily per process (_ensure_open) and excluded from
        # pickling: an open np.memmap pickles as a plain ndarray, i.e. spawn
        # dataloader workers would serialise the full cube array.
        self._cube = self._rgb = self._mag = self._gt = None
        shape0 = np.load(self._paths['cube'], mmap_mode='r').shape[0]
        assert len(self.event_ids) == shape0

        if split == 'train':
            df = pd.read_csv(os.path.join(root_dir, windows_csv),
                             usecols=['event_id', 'has_plume',
                                      'window_row_off', 'window_col_off',
                                      'window_width', 'window_height'],
                             low_memory=False)
            eid2idx = {e: i for i, e in enumerate(self.event_ids)}
            keep = df['event_id'].astype(str).map(eid2idx)
            missing = int(keep.isna().sum())
            if missing:
                print(f'[EMITGridDataset] dropping {missing} windows whose tile '
                      f'is missing from the store')
                df = df[~keep.isna()]
                keep = keep.dropna()
            self.win_tile = keep.to_numpy(dtype=np.int64)
            self.win_row = df['window_row_off'].to_numpy(dtype=np.int64)
            self.win_col = df['window_col_off'].to_numpy(dtype=np.int64)
            self.win_plume = df['has_plume'].to_numpy(dtype=bool)
            assert int(df['window_width'].max()) == self.patch_size
            if max_windows is not None:
                self.win_tile = self.win_tile[:max_windows]
                self.win_row = self.win_row[:max_windows]
                self.win_col = self.win_col[:max_windows]
                self.win_plume = self.win_plume[:max_windows]
            n_p = int(self.win_plume.sum())
            print(f'[EMITGridDataset:train] {len(self.win_tile):,} windows '
                  f'({n_p:,} plume = {n_p / len(self.win_tile):.2%})')
        else:
            self.n_tiles = len(self.event_ids)
            if max_tiles is not None:
                self.n_tiles = min(self.n_tiles, max_tiles)

    # Plume windows get 1/plume_fraction, others 1/(1-plume_fraction) - each
    # group sums to N, so a with-replacement draw is 50:50 plume/non-plume in
    # expectation.
    @property
    def sample_weights(self) -> np.ndarray:
        assert self.split == 'train'
        frac = self.win_plume.mean()
        w = np.where(self.win_plume, 1.0 / frac, 1.0 / (1.0 - frac))
        return w.astype(np.float64)

    def __len__(self):
        return len(self.win_tile) if self.split == 'train' else self.n_tiles

    def _ensure_open(self):
        if self._cube is None:
            self._cube = np.load(self._paths['cube'], mmap_mode='r')
            self._rgb = np.load(self._paths['rgb'], mmap_mode='r')
            self._mag = np.load(self._paths['mag'], mmap_mode='r')
            self._gt = np.load(self._paths['gt'], mmap_mode='r')

    def get_full_tile(self, tile_idx: int):
        """Full 512x512 tile tensors by STORE index (both splits) - used by the
        training-time visualization hook; read-only."""
        self._ensure_open()
        cube = np.asarray(self._cube[tile_idx], dtype=np.float32) * self.CUBE_SCALE
        rgb = np.asarray(self._rgb[tile_idx], dtype=np.float32)
        mag = np.asarray(self._mag[tile_idx], dtype=np.float32)
        gt = np.asarray(self._gt[tile_idx], dtype=np.float32)
        return (torch.from_numpy(cube), torch.from_numpy(rgb),
                torch.from_numpy(gt), torch.from_numpy(mag))

    def tile_gt_sums(self) -> np.ndarray:
        """Per-tile GT pixel counts over the whole store (vis tile selection)."""
        self._ensure_open()
        return np.asarray(self._gt, dtype=np.uint8).reshape(self._gt.shape[0], -1).sum(1)

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ('_cube', '_rgb', '_mag', '_gt'):
            state[k] = None
        return state

    @staticmethod
    def _augment_arrays(cube, rgb, gt, mag):
        if random.random() > 0.5:
            cube = cube[..., ::-1]; rgb = rgb[..., ::-1]
            gt = gt[..., ::-1]; mag = mag[..., ::-1]
        if random.random() > 0.5:
            cube = cube[..., ::-1, :]; rgb = rgb[..., ::-1, :]
            gt = gt[..., ::-1, :]; mag = mag[..., ::-1, :]
        k = random.randint(0, 3)
        if k:
            cube = np.rot90(cube, k, (-2, -1)); rgb = np.rot90(rgb, k, (-2, -1))
            gt = np.rot90(gt, k, (-2, -1)); mag = np.rot90(mag, k, (-2, -1))
        return cube, rgb, gt, mag

    def __getitem__(self, idx: int):
        self._ensure_open()
        if self.split == 'train':
            t, r, c = int(self.win_tile[idx]), int(self.win_row[idx]), int(self.win_col[idx])
            ps = self.patch_size
            cube = np.asarray(self._cube[t, :, r:r + ps, c:c + ps], dtype=np.float32)
            rgb = np.asarray(self._rgb[t, :, r:r + ps, c:c + ps], dtype=np.float32)
            mag = np.asarray(self._mag[t, r:r + ps, c:c + ps], dtype=np.float32)
            gt = np.asarray(self._gt[t, r:r + ps, c:c + ps], dtype=np.float32)
            if self.augment:
                cube, rgb, gt, mag = self._augment_arrays(cube, rgb, gt, mag)
        else:
            cube = np.asarray(self._cube[idx], dtype=np.float32)
            rgb = np.asarray(self._rgb[idx], dtype=np.float32)
            mag = np.asarray(self._mag[idx], dtype=np.float32)
            gt = np.asarray(self._gt[idx], dtype=np.float32)
        cube = np.ascontiguousarray(cube) * self.CUBE_SCALE
        return (
            torch.from_numpy(cube),
            torch.from_numpy(np.ascontiguousarray(rgb)),
            torch.from_numpy(np.ascontiguousarray(gt)),
            torch.from_numpy(np.ascontiguousarray(mag)),
        )


class DistributedWeightedSampler(Sampler):
    """Per-rank i.i.d. with-replacement weighted sampling (weighted sampler under DDP).

    With replacement=True a WeightedRandomSampler is i.i.d., so independent
    per-rank draws of N/world_size samples are statistically identical to one
    global draw of N. Seeds differ per (seed, epoch, rank).
    """

    def __init__(self, weights, num_samples: int, seed: int = 42, rank: int = 0):
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.double)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch * 1009 + self.rank)
        idx = torch.multinomial(self.weights, self.num_samples,
                                replacement=True, generator=g)
        return iter(idx.tolist())


def load_difficulty_map(root: str, csv_name: str = 'test_filtered_v2.csv'):
    """Map event_id -> {'easy', 'hard', 'none'} from the test CSV ``source_label``.

    Plume-free tiles (has_plume False / unparseable source_label) map to 'none'.
    The ``difficulty`` flag is the STARCOP-inherited label embedded in the
    ``source_label`` JSON of each plume row.
    """
    import ast
    df = pd.read_csv(os.path.join(root, csv_name),
                     usecols=['event_id', 'has_plume', 'source_label'],
                     low_memory=False)
    out = {}
    for _, row in df.iterrows():
        eid = str(row['event_id'])
        diff = 'none'
        if bool(row['has_plume']):
            try:
                d = ast.literal_eval(str(row['source_label']))
                if isinstance(d, dict):
                    diff = str(d.get('difficulty', 'none')).lower()
            except (ValueError, SyntaxError):
                diff = 'none'
        out[eid] = diff
    return out
