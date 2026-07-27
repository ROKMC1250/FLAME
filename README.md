# FLAME: Physics-Guided Neural Operators for Onboard Satellite Methane Detection in Hyperspectral Imagery

Official code for FLAME, a lightweight physics-guided methane plume
segmentation model. A Fourier neural
operator (FNO / U-FNO) backbone with channel squeeze-and-excitation produces a
shared feature map; two heads estimate the per-pixel background spectrum and a
per-band spectral weight, which are combined with a fixed CH4 absorption
spectrum into a **parameter-free, physics-based score map**; a small
convolutional head segments plumes from the concatenation of backbone
features, the score map, and RGB.

The same codebase trains and evaluates FLAME on two benchmarks, selected by a
single `dataset:` key in the config — each key switches the full protocol
(sampling regime, loss schedule, normalisation, and evaluation):

| | `dataset: starcop` | `dataset: emit` |
|---|---|---|
| Sensor / data | AVIRIS-NG ([STARCOP](https://github.com/spaceml-org/STARCOP)) | EMIT (synthetic; [OxHyperSyntheticCH4](https://huggingface.co/datasets/previtus/OxHyperSyntheticCH4)) |
| SWIR input | 72 bands, 2122–2488 nm | 64 bands, 2004–2478 nm |
| Training | full 512×512 tiles, two-phase curriculum (aux score pretrain → seg + decaying aux) | 64×64 / stride-32 grid windows, 50:50 plume-balanced sampler, BCE, bf16 |
| Seg-head score input | physics score (self-contained) | physics score; mag1c as auxiliary target (`use_mag_in_seg: true` switches to mag1c input) |
| Evaluation | full-tile, sigmoid > 0.5 + 3×3 morph. opening; pixel F1/IoU, strong/weak subsets, tile FPR | tile-level 64/32 windows, logits ≥ 0, no morph.; F1/IoU/AUPRC, easy/hard subsets (protocol of [HyperspectralViTs](https://arxiv.org/abs/2410.17248)) |

## Installation

```bash
conda create -n flame python=3.10 -y
conda activate flame
pip install -r requirements.txt
```

PyTorch with CUDA is required for training; evaluation also runs on CPU.

## Pretrained weights

Two checkpoints are released on the Hugging Face Hub
([hjh1037/FLAME](https://huggingface.co/hjh1037/FLAME)), one per benchmark;
additional seeds are available under `seeds/` for multi-seed reproduction.

| Weights | Trained on | Config |
|---|---|---|
| `flame_starcop.pt` | STARCOP (AVIRIS-NG) | `configs/flame_starcop.yaml` |
| `flame_emit.pt` | OxHyperSyntheticCH4 (EMIT) | `configs/flame_emit.yaml` |

```bash
pip install huggingface_hub
hf download hjh1037/FLAME flame_starcop.pt flame_emit.pt --local-dir .
```

Evaluation and visualization discover runs under
`logs/<uid>/seed_<N>/weights/best.pt` with a frozen `config.yaml` beside the
`weights/` directory. Place the downloaded checkpoints accordingly:

```bash
mkdir -p logs/flame_starcop/seed_42/weights logs/flame_emit/seed_42/weights
cp flame_starcop.pt logs/flame_starcop/seed_42/weights/best.pt
cp configs/flame_starcop.yaml logs/flame_starcop/seed_42/config.yaml
cp flame_emit.pt logs/flame_emit/seed_42/weights/best.pt
cp configs/flame_emit.yaml logs/flame_emit/seed_42/config.yaml
```

Resulting layout (training runs produce the same structure automatically):

```
logs/
├── flame_starcop/
│   └── seed_42/
│       ├── config.yaml          # frozen config (read by evaluate.py / visualize.py)
│       └── weights/best.pt
└── flame_emit/
    └── seed_42/
        ├── config.yaml
        └── weights/best.pt
```

## Data preparation

### STARCOP (AVIRIS-NG)

```bash
python scripts/download_starcop.py               # full dataset, ~633 GB
python scripts/download_starcop.py --eval-only   # evaluation split only, ~82 GB
```

Downloads the four all-bands Hugging Face dataset repos
(`previtus/STARCOP_allbands_{Train1,Train2,Train3,Eval}`; see the
[STARCOP project](https://github.com/spaceml-org/STARCOP)) and arranges them
under `datasets/starcop/` — each tile a directory of per-band
`TOA_AVIRIS_{wavelength}nm.tif` files plus `labelbinary.tif`, split CSVs at
the root, and the evaluation tiles both under `STARCOP_allbands_Eval/` and
hard-linked at the root:

```
datasets/starcop/
├── test.csv
├── STARCOP_allbands_Eval/<tile_id>/...
└── <tile_id>/
    ├── TOA_AVIRIS_2122nm.tif ... TOA_AVIRIS_2488nm.tif
    ├── TOA_AVIRIS_640nm.tif / 550nm / 460nm      # RGB
    └── labelbinary.tif
```

Training (not evaluation) additionally requires a per-tile
`mag1c_sas_cache.npy` — the mag1c matched filter run with 1% covariance
subsampling (`mag1c --sample 0.01` on the 2122–2488 nm bands), used as the
auxiliary target of the physics score head. One command generates and imports
them (clones the
[methane filters benchmark](https://github.com/zaitra/methane-filters-benchmark)
on first use; requires a CUDA GPU and `pip install pysptools imagecodecs`;
roughly 3 s per tile):

```bash
python scripts/generate_mag1c_products.py --root datasets/starcop --csv train.csv
```

If you already have benchmark products, import them directly with
`scripts/import_mag1c_products.py`.

Optional speed-up: pre-stack the SWIR bands and point `data.npy_dir` at the
cache.

```bash
python scripts/build_starcop_npy_cache.py --root datasets/starcop --out datasets/starcop_swir_npy
```

### EMIT (OxHyperSyntheticCH4)

```bash
python scripts/download_emit.py
```

Downloads [OxHyperSyntheticCH4](https://huggingface.co/datasets/previtus/OxHyperSyntheticCH4)
to `datasets/oxhyper_synthetic_ch4/` (per-event directories plus the split
CSVs `train_filtered_v2.csv`, `val_filtered_v2.csv`, `test_filtered_v2.csv`,
and the window index `train_filtered_v2_tiled_64_32.csv`). Each event
directory contains the ENVI cube `B`/`B.hdr`, RGB TIFFs, the mag1c product
`B_magic30_tile.tif` (precomputed — no mag1c generation needed on EMIT), and
`labelbinary.tif`.

The first training run consolidates each split into a flat fp16 memmap store
under `data.store_dir` (~30 GB for train); subsequent runs reuse it. Optional
speed-up for the store build:

```bash
python scripts/build_emit_npy_cache.py --root datasets/oxhyper_synthetic_ch4 --out datasets/emit_npy64
```

## Training

```bash
# STARCOP — DDP over all visible GPUs
python train.py --config configs/flame_starcop.yaml --seed 42

# EMIT
python train.py --config configs/flame_emit.yaml --seed 42

# Debug: single GPU, no checkpoints, tiny run
python train.py --config configs/flame_emit.yaml --uid dbg --no-ddp --no-save --smoke
```

Common flags: `--uid --seed --lr --model-path --no-ddp --no-save --smoke
--port`. `train.batch_size` must be divisible by the number of GPUs.
Checkpoints, the frozen config, and TensorBoard events are written to
`logs/<uid>/seed_<seed>/`; re-running the same uid with a different `--seed`
adds a sibling `seed_<M>/` directory.

For multi-seed reporting:

```bash
for s in 42 43 44; do python train.py --config configs/flame_starcop.yaml --seed $s; done
python evaluate.py --uid flame_starcop     # reports mean ± std over seeds
```

## Evaluation

```bash
python evaluate.py --uid flame_starcop     # -> results/starcop/metrics.{csv,md}
python evaluate.py --uid flame_emit        # -> results/emit/metrics.{csv,md}
```

The protocol is selected by the `dataset` key of the run's frozen config (or
forced with `--dataset starcop|emit`). All seeds of a uid are evaluated and
aggregated to mean ± std. Protocol-specific flags (`--root`, `--threshold`,
`--eval-mode`, ...) are listed by `python evaluate.py --dataset emit --help`.

## Visualization

```bash
python visualize.py --uid flame_starcop --max-tiles 20
python visualize.py --uid flame_emit --all-tiles --max-tiles 40
```

Writes per-tile panel figures (RGB | score/mag1c | probability | prediction |
GT | error map) to `logs/<uid>/seed_<N>/visualizations/`.

## Repository layout

```
flame/
├── model.py             # FLAME (SpectralConv2d, U-FNO layers, heads)
├── datasets/            # starcop.py, emit.py (grid windows + RAM store)
├── trainers/            # base.py, starcop.py (curriculum), emit.py (grid regime)
├── train_starcop.py     # per-dataset launchers (called by train.py)
├── train_emit.py
├── eval_starcop.py      # per-dataset protocols (called by evaluate.py)
├── eval_emit.py
├── metrics.py           # shared pixel metrics + AUPRC
└── vis.py               # panel figures
configs/                 # flame_starcop.yaml, flame_emit.yaml
resources/               # CH4 spectra, band centers, background statistics
scripts/                 # data caches + mag1c product import
```

## License

Apache-2.0 — see [LICENSE](LICENSE). The STARCOP and OxHyperSyntheticCH4 datasets and
the mag1c / methane-filters-benchmark tools are distributed under their own
licenses.

## Citation

```bibtex
@article{flame2026,
  title   = {FLAME: Physics-Guided Neural Operators for Onboard Satellite
             Methane Detection in Hyperspectral Imagery},
  author  = {},
  year    = {2026},
  note    = {Citation to be updated upon publication.}
}
```
