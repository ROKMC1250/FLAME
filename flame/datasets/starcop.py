"""STARCOP (AVIRIS-NG) dataset for FLAME.

Tile layout: per-band ``TOA_AVIRIS_{wv}nm.tif`` + ``labelbinary.tif`` (GT).
Native 512x512; smaller tiles are zero-padded (padding is excluded from the
loss/metrics via the valid mask ``cube.abs().sum(0) > 0``).

SWIR bands: 72 bands in 2122-2488 nm, auto-discovered per tile.
Optional fast paths: ``npy_dir`` (pre-stacked SWIR-only .npy, fp16 or fp32 —
see scripts/build_starcop_npy_cache.py) or ``npz_dir``.

``STARCOPTrainDataset`` additionally serves the per-tile mag1c-sas score cache
(``mag1c_sas_cache.npy``) used as the auxiliary target of the physics score
head — see scripts/import_mag1c_products.py.
"""
import os
import random

import numpy as np
import tifffile as tiff
import torch

RGB_WAVELENGTHS = [640, 550, 460]  # R, G, B


class STARCOPDataset(torch.utils.data.Dataset):
    """STARCOP hyperspectral tile dataset.

    Loads SWIR bands (2122-2488 nm, 72 bands) and binary ground-truth labels.
    Training mode excludes tiles present in the validation set.
    """

    def __init__(self, train_data_path, val_data_path=None, wv_range=(2122, 2488),
                 validation=False, val_max_tiles=None, tile_size=None,
                 npz_dir=None, npy_dir=None, load_rgb=False, load_swir=True,
                 augment=False, crop_size=None, cache_small_files=False):
        super().__init__()
        self.wv_range = tuple(wv_range) if not isinstance(wv_range, tuple) else wv_range
        self.validation = validation
        self.npz_dir = npz_dir
        self.npy_dir = npy_dir
        self.load_rgb = load_rgb
        self.load_swir = load_swir
        self.augment = augment and not validation  # only augment training
        self.crop_size = crop_size if not validation else None  # only crop training

        # Normalize train_data_path to a list
        if isinstance(train_data_path, str):
            train_data_paths = [train_data_path]
        else:
            train_data_paths = list(train_data_path)

        if validation:
            assert val_data_path is not None, "val_data_path required for validation"
            self.tile_paths = self._discover_tiles_from_dirs([val_data_path], exclude=None)
            if val_max_tiles and len(self.tile_paths) > val_max_tiles:
                self.tile_paths = self.tile_paths[:val_max_tiles]
        else:
            # Exclude tiles that are in the validation set
            exclude = set()
            if val_data_path and os.path.isdir(val_data_path):
                exclude = set(
                    d for d in os.listdir(val_data_path)
                    if os.path.isdir(os.path.join(val_data_path, d))
                )
            self.tile_paths = self._discover_tiles_from_dirs(train_data_paths, exclude=exclude)

        # Pad smaller tiles to tile_size for batching (zero-pad; zeros are masked)
        self.pad_to = tile_size

        self.tile_ids = [os.path.basename(p) for p in self.tile_paths]

        # Determine SWIR band indices
        if npy_dir:
            # npy files already contain SWIR-only bands — no index needed
            self.selected_wavelengths = self._select_wavelengths(self.tile_paths[0])
        elif npz_dir:
            self._init_npz_band_indices()
        else:
            self.selected_wavelengths = self._select_wavelengths(self.tile_paths[0])

        # Cache GT and RGB in memory to eliminate per-sample tiff I/O
        self._gt_cache = None
        self._rgb_cache = None
        if cache_small_files:
            self._gt_cache = []
            if load_rgb:
                self._rgb_cache = []
            for tp in self.tile_paths:
                gt = tiff.imread(os.path.join(tp, 'labelbinary.tif'))
                if gt.ndim == 3:
                    gt = gt[0]
                self._gt_cache.append(gt.astype(np.float32))
                if load_rgb:
                    rgb_bands = [tiff.imread(os.path.join(tp, f'TOA_AVIRIS_{wv}nm.tif')).astype(np.float32)
                                 for wv in RGB_WAVELENGTHS]
                    self._rgb_cache.append(np.stack(rgb_bands, axis=0))

    def _init_npz_band_indices(self):
        """Precompute SWIR band indices from the first npz file."""
        sample_npz = os.path.join(self.npz_dir, self.tile_ids[0] + '.npz')
        with np.load(sample_npz) as data:
            wvs = data['wavelengths']
        mask = (wvs >= self.wv_range[0]) & (wvs <= self.wv_range[1])
        self.npz_band_indices = np.where(mask)[0]
        self.selected_wavelengths = sorted(wvs[mask].astype(int).tolist())

    @staticmethod
    def _discover_tiles_from_dirs(data_paths, exclude=None):
        """Find tile directories containing labelbinary.tif across multiple data dirs."""
        exclude = exclude or set()
        tile_paths = []
        for data_path in data_paths:
            for name in sorted(os.listdir(data_path)):
                tile_dir = os.path.join(data_path, name)
                gt_path = os.path.join(tile_dir, 'labelbinary.tif')
                if os.path.isdir(tile_dir) and os.path.exists(gt_path) and name not in exclude:
                    tile_paths.append(tile_dir)
        return tile_paths

    def _select_wavelengths(self, tile_path):
        """Scan a tile folder and select wavelengths within wv_range."""
        files = sorted(os.listdir(tile_path))
        band_files = [f for f in files if f.startswith('TOA_AVIRIS_') and f.endswith('nm.tif')]
        all_wv = [int(f.replace('TOA_AVIRIS_', '').replace('nm.tif', '')) for f in band_files]
        selected = sorted([w for w in all_wv if self.wv_range[0] <= w <= self.wv_range[1]])
        return selected

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        tile_id = self.tile_ids[idx]

        if not self.load_swir:
            # Skip the 72-band SWIR load entirely. Placeholder cube is filled
            # post-GT-load so _pad/_random_crop can still index spatial dims.
            cube = None
        elif self.npy_dir:
            # Fastest path: SWIR-only npy (B, H, W); fp16 files get promoted once.
            npy_path = os.path.join(self.npy_dir, tile_id + '.npy')
            cube = np.load(npy_path).astype(np.float32, copy=False)
        elif self.npz_dir:
            npz_path = os.path.join(self.npz_dir, tile_id + '.npz')
            with np.load(npz_path) as data:
                cube = data['cube'][:, :, self.npz_band_indices]  # (H, W, B_swir)
            cube = cube.transpose(2, 0, 1)  # (B, H, W)
        else:
            # Slow path: 72 separate tif reads
            bands = []
            for wv in self.selected_wavelengths:
                fpath = os.path.join(tile_path, f'TOA_AVIRIS_{wv}nm.tif')
                bands.append(tiff.imread(fpath))
            cube = np.stack(bands, axis=0).astype(np.float32)  # (B, H, W)

        # Load ground truth
        if self._gt_cache is not None:
            gt = self._gt_cache[idx]
        else:
            gt = tiff.imread(os.path.join(tile_path, 'labelbinary.tif'))
            if gt.ndim == 3:
                gt = gt[0]
            gt = gt.astype(np.float32)

        # Placeholder cube shares spatial dims with gt so downstream
        # pad/crop/augment paths (which key off cube.shape[-2:]) keep working.
        if cube is None:
            cube = np.zeros((1, gt.shape[0], gt.shape[1]), dtype=np.float32)

        if self.load_rgb:
            if self._rgb_cache is not None:
                rgb = self._rgb_cache[idx]
            else:
                rgb_bands = []
                for wv in RGB_WAVELENGTHS:
                    fpath = os.path.join(tile_path, f'TOA_AVIRIS_{wv}nm.tif')
                    rgb_bands.append(tiff.imread(fpath).astype(np.float32))
                rgb = np.stack(rgb_bands, axis=0)  # (3, H, W)
            cube, rgb, gt = torch.from_numpy(cube), torch.from_numpy(rgb), torch.from_numpy(gt)
            if self.augment:
                cube, rgb, gt = self._augment(cube, rgb, gt)
            cube, rgb, gt = self._pad(cube, rgb, gt)
            cube, rgb, gt = self._random_crop(cube, rgb, gt)
            return cube, rgb, gt

        cube, gt = torch.from_numpy(cube), torch.from_numpy(gt)
        if self.augment:
            cube, _, gt = self._augment(cube, None, gt)
        cube, _, gt = self._pad(cube, None, gt)
        cube, _, gt = self._random_crop(cube, None, gt)
        return cube, gt

    def _pad(self, cube, rgb, gt):
        """Zero-pad smaller tiles to self.pad_to size for uniform batching."""
        if self.pad_to is None:
            return cube, rgb, gt
        H, W = cube.shape[-2], cube.shape[-1]
        if H >= self.pad_to and W >= self.pad_to:
            return cube, rgb, gt
        pH, pW = max(self.pad_to - H, 0), max(self.pad_to - W, 0)
        cube = torch.nn.functional.pad(cube, (0, pW, 0, pH))  # right, bottom
        gt = torch.nn.functional.pad(gt, (0, pW, 0, pH))
        if rgb is not None:
            rgb = torch.nn.functional.pad(rgb, (0, pW, 0, pH))
        return cube, rgb, gt

    def _random_crop(self, cube, rgb, gt):
        """Random spatial crop for training. No-op if crop_size is None."""
        if self.crop_size is None:
            return cube, rgb, gt
        H, W = cube.shape[-2], cube.shape[-1]
        if H <= self.crop_size or W <= self.crop_size:
            return cube, rgb, gt
        y = random.randint(0, H - self.crop_size)
        x = random.randint(0, W - self.crop_size)
        cube = cube[:, y:y + self.crop_size, x:x + self.crop_size]
        gt = gt[y:y + self.crop_size, x:x + self.crop_size]
        if rgb is not None:
            rgb = rgb[:, y:y + self.crop_size, x:x + self.crop_size]
        return cube, rgb, gt

    @staticmethod
    def _augment(cube, rgb, gt):
        """Random spatial augmentation (flips + 90-degree rotation)."""
        if random.random() > 0.5:
            cube = torch.flip(cube, [-1])  # horizontal flip
            if rgb is not None:
                rgb = torch.flip(rgb, [-1])
            gt = torch.flip(gt, [-1])
        if random.random() > 0.5:
            cube = torch.flip(cube, [-2])  # vertical flip
            if rgb is not None:
                rgb = torch.flip(rgb, [-2])
            gt = torch.flip(gt, [-2])
        k = random.randint(0, 3)  # 0/90/180/270 rotation
        if k > 0:
            cube = torch.rot90(cube, k, [-2, -1])
            if rgb is not None:
                rgb = torch.rot90(rgb, k, [-2, -1])
            gt = torch.rot90(gt, k, [-2, -1])
        return cube, rgb, gt


class STARCOPTrainDataset(torch.utils.data.Dataset):
    """Wraps STARCOPDataset to add the mag1c-sas score cache as 4th element.

    Augmentation is applied here (not in the base dataset) so that all four
    tensors (cube, rgb, gt, mag) receive the same spatial transform.
    """

    def __init__(self, base_ds, score_divisor=1750.0, augment=False):
        self.base = base_ds
        self.score_divisor = score_divisor
        self.tile_paths = base_ds.tile_paths
        self.augment = augment
        # Pre-load mag1c caches as a SINGLE contiguous array. A Python list of
        # N separate numpy arrays would create N distinct PyObjects; DataLoader
        # worker forks then break COW on every refcount touch, ballooning
        # per-worker RSS. One big ndarray = one PyObject = COW stays intact.
        mags = []
        for tp in base_ds.tile_paths:
            mag = np.load(os.path.join(tp, 'mag1c_sas_cache.npy')).astype(np.float32)
            if base_ds.pad_to is not None:
                H, W = mag.shape
                pH, pW = max(base_ds.pad_to - H, 0), max(base_ds.pad_to - W, 0)
                if pH > 0 or pW > 0:
                    mag = np.pad(mag, ((0, pH), (0, pW)))
            mag = np.clip(mag / score_divisor, 0, 2.0)
            mags.append(mag)
        self._mag_cache = np.stack(mags, axis=0)  # (N, H, W) contiguous float32

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        cube, rgb, gt = self.base[idx]
        # .copy() detaches this slice from the shared cache so the returned
        # tensor doesn't hold a view on the big array across the DataLoader
        # worker -> main process pipe.
        mag_norm = torch.from_numpy(self._mag_cache[idx].copy())
        if self.augment:
            cube, rgb, gt, mag_norm = self._augment4(cube, rgb, gt, mag_norm)
        return cube, rgb, gt, mag_norm

    @staticmethod
    def _augment4(cube, rgb, gt, mag):
        if random.random() > 0.5:
            cube = torch.flip(cube, [-1])
            rgb = torch.flip(rgb, [-1])
            gt = torch.flip(gt, [-1])
            mag = torch.flip(mag, [-1])
        if random.random() > 0.5:
            cube = torch.flip(cube, [-2])
            rgb = torch.flip(rgb, [-2])
            gt = torch.flip(gt, [-2])
            mag = torch.flip(mag, [-2])
        k = random.randint(0, 3)
        if k > 0:
            cube = torch.rot90(cube, k, [-2, -1])
            rgb = torch.rot90(rgb, k, [-2, -1])
            gt = torch.rot90(gt, k, [-2, -1])
            mag = torch.rot90(mag, k, [-2, -1])
        return cube, rgb, gt, mag


def load_tile_for_inference(tile_path, npy_dir=None, selected_wv=None,
                            wv_range=(2122, 2488), load_rgb=True, pad_to=512):
    """Load a single STARCOP tile (no augmentation) for inference/visualization.

    Returns:
        cube: (B, H, W) tensor — SWIR bands (padded to pad_to)
        rgb:  (3, H, W) tensor or None
        gt:   (H, W) tensor
        (H, W): original spatial dims before padding
    """
    tile_id = os.path.basename(tile_path)

    if npy_dir:
        cube = np.load(os.path.join(npy_dir, tile_id + '.npy')).astype(np.float32, copy=False)
    else:
        if selected_wv is None:
            files = sorted(os.listdir(tile_path))
            band_files = [f for f in files
                          if f.startswith('TOA_AVIRIS_') and f.endswith('nm.tif')]
            all_wv = sorted(int(f.replace('TOA_AVIRIS_', '').replace('nm.tif', ''))
                            for f in band_files)
            selected_wv = [w for w in all_wv if wv_range[0] <= w <= wv_range[1]]
        bands = [tiff.imread(os.path.join(tile_path, f'TOA_AVIRIS_{wv}nm.tif'))
                 for wv in selected_wv]
        cube = np.stack(bands, axis=0).astype(np.float32)

    gt = tiff.imread(os.path.join(tile_path, 'labelbinary.tif')).astype(np.float32)
    if gt.ndim == 3:
        gt = gt[0]

    cube = torch.from_numpy(cube)
    gt = torch.from_numpy(gt)

    rgb = None
    if load_rgb:
        rgb_bands = [tiff.imread(os.path.join(tile_path, f'TOA_AVIRIS_{wv}nm.tif')).astype(np.float32)
                     for wv in RGB_WAVELENGTHS]
        rgb = torch.from_numpy(np.stack(rgb_bands, axis=0))

    H, W = cube.shape[-2], cube.shape[-1]
    if pad_to is not None:
        pH, pW = max(pad_to - H, 0), max(pad_to - W, 0)
        if pH > 0 or pW > 0:
            cube = torch.nn.functional.pad(cube, (0, pW, 0, pH))
            gt = torch.nn.functional.pad(gt, (0, pW, 0, pH))
            if rgb is not None:
                rgb = torch.nn.functional.pad(rgb, (0, pW, 0, pH))

    return cube, rgb, gt, (H, W)
