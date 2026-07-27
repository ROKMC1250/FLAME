# FLAME: Physics-Guided Neural Operators for Onboard Satellite Methane Detection in Hyperspectral Imagery

This repository is the official implementation of
[FLAME: Physics-Guided Neural Operators for Onboard Satellite Methane Detection in Hyperspectral Imagery](https://arxiv.org/abs/2606.01577).
Pretrained weights are available at [hjh1037/FLAME](https://huggingface.co/hjh1037/FLAME).

Methane is a major driver of near-term climate change, and rapidly identifying
its emission sources is a critical climate intervention. Spaceborne
hyperspectral imagery is the primary tool for this task, but the volume of
data produced by each sensor makes ground-based detection impractical and
necessitates onboard detection. Classical methods incur prohibitive
computational cost on onboard hardware, while deep learning models are fast
but fall short on detection quality. FLAME is a physics-guided neural operator
that builds the physics of methane absorption directly into its architecture.
On the STARCOP benchmark, FLAME achieves the highest detection accuracy among
all evaluated methods, reduces the pixel-level false positive rate by nearly
3x over the strongest neural baseline, uses the fewest parameters among
learned baselines, and runs within the latency budget of onboard satellite
hardware. Beyond the paper, this repository also includes experiments on the
EMIT OxHyperSyntheticCH4 dataset, where FLAME is trained and evaluated under
the same protocol as HyperspectralViTs.

<p align="center">
  <img src="assets/flame_overview.png" width="55%" alt="FLAME overview">
</p>

## Results

### STARCOP

Mean over three seeds. Standard deviations and inference times are reported
in the paper. Pixel FPR is x10^-4.

| Method | F1 | IoU | Precision | Recall | Pixel FPR | Params |
|---|---|---|---|---|---|---|
| CEM | 0.178 | 0.098 | 0.114 | 0.399 | 74.0 | - |
| MF | 0.178 | 0.098 | 0.115 | 0.398 | 74.0 | - |
| ACE | 0.154 | 0.084 | 0.111 | 0.254 | 49.0 | - |
| MAG1C-tile | 0.300 | 0.176 | 0.228 | 0.437 | 35.0 | - |
| MAG1C-SAS | 0.284 | 0.166 | 0.194 | 0.528 | 52.0 | - |
| UNet+MAG1C-tile | 0.477 | 0.314 | 0.320 | 0.946 | 49.0 | 6.60M |
| UNet+MAG1C-SAS | 0.441 | 0.283 | 0.300 | 0.844 | 48.0 | 6.60M |
| LinkNet+MAG1C-tile | 0.402 | 0.252 | 0.256 | 0.947 | 67.0 | 0.85M |
| LinkNet+MAG1C-SAS | 0.397 | 0.248 | 0.260 | 0.846 | 58.0 | 0.85M |
| UNet MNv2 | 0.421 | 0.281 | 0.337 | 0.570 | 27.0 | 6.65M |
| UNet R18 | 0.309 | 0.183 | 0.218 | 0.535 | 46.0 | 14.54M |
| SegFormer | 0.393 | 0.246 | 0.286 | 0.736 | 52.0 | 3.82M |
| SegFormer CU | 0.515 | 0.347 | 0.417 | 0.679 | 23.0 | 4.30M |
| SegFormer CUS | 0.449 | 0.299 | 0.314 | 0.831 | 46.0 | 4.30M |
| EfficientViT | 0.345 | 0.209 | 0.222 | 0.785 | 66.0 | 4.81M |
| EfficientViT CU | 0.404 | 0.253 | 0.262 | 0.880 | 59.0 | 4.85M |
| EfficientViT CUS | 0.414 | 0.264 | 0.279 | 0.829 | 55.0 | 4.85M |
| **FLAME** | **0.608** | **0.437** | **0.651** | 0.576 | **8.0** | **0.78M** |

<p align="center">
  <img src="assets/qualitative_starcop.png" width="95%" alt="Qualitative comparison on STARCOP">
</p>

### EMIT

Test tiles are scored as 64x64 windows at stride 32 following the
HyperspectralViTs protocol. Learned baselines are our reproductions. Mean over
three runs.

| Method | F1 | Precision | Recall | AUPRC |
|---|---|---|---|---|
| ACE | 0.031 | 0.020 | 0.075 | 0.006 |
| MF | 0.053 | 0.034 | 0.127 | 0.026 |
| CEM | 0.183 | 0.116 | 0.437 | 0.281 |
| MAG1C | 0.215 | 0.136 | 0.514 | 0.208 |
| EfficientViT | 0.389 | 0.352 | 0.436 | 0.385 |
| SegFormer | 0.430 | 0.383 | 0.543 | 0.479 |
| EfficientViT CUS | 0.448 | 0.342 | 0.660 | 0.570 |
| EfficientViT CU | 0.467 | 0.410 | 0.554 | 0.495 |
| UNet MF+RGB | 0.551 | 0.518 | 0.593 | 0.580 |
| SegFormer CU | 0.576 | 0.519 | 0.679 | 0.647 |
| SegFormer CUS | 0.619 | 0.521 | 0.766 | 0.740 |
| **FLAME** | **0.798** | **0.805** | **0.791** | **0.837** |

## Installation

```bash
conda create -n flame python=3.10 -y
conda activate flame
pip install -r requirements.txt
```

## Pretrained weights

| Weights | Trained on | Config |
|---|---|---|
| `flame_starcop.pt` | STARCOP | `configs/flame_starcop.yaml` |
| `flame_emit.pt` | OxHyperSyntheticCH4 | `configs/flame_emit.yaml` |

```bash
pip install huggingface_hub
hf download hjh1037/FLAME flame_starcop.pt flame_emit.pt --local-dir .

mkdir -p logs/flame_starcop/seed_42/weights logs/flame_emit/seed_42/weights
cp flame_starcop.pt logs/flame_starcop/seed_42/weights/best.pt
cp configs/flame_starcop.yaml logs/flame_starcop/seed_42/config.yaml
cp flame_emit.pt logs/flame_emit/seed_42/weights/best.pt
cp configs/flame_emit.yaml logs/flame_emit/seed_42/config.yaml
```

`evaluate.py` and `visualize.py` discover checkpoints under
`logs/<uid>/seed_42/weights/best.pt` with a `config.yaml` beside the
`weights` directory, which is the same layout that training produces.

## Data preparation

### STARCOP

STARCOP was released by Ruzicka et al. in Scientific Reports 2023. We use the
125-band version distributed by the authors on the Hugging Face Hub:
[Train1](https://huggingface.co/datasets/previtus/STARCOP_allbands_Train1),
[Train2](https://huggingface.co/datasets/previtus/STARCOP_allbands_Train2),
[Train3](https://huggingface.co/datasets/previtus/STARCOP_allbands_Train3),
[Eval](https://huggingface.co/datasets/previtus/STARCOP_allbands_Eval).

The following script downloads all four repositories and arranges them under
`datasets/starcop/`:

```bash
python scripts/download_starcop.py               # full dataset, ~633 GB
python scripts/download_starcop.py --eval-only   # evaluation split only, ~82 GB
```

After the script finishes the layout is:

```
datasets/starcop/
├── test.csv
├── train.csv
├── STARCOP_allbands_Eval/<tile_id>/...
└── <tile_id>/
    ├── TOA_AVIRIS_2122nm.tif ... TOA_AVIRIS_2488nm.tif
    ├── TOA_AVIRIS_640nm.tif / 550nm / 460nm
    └── labelbinary.tif
```

Training additionally needs a per-tile MAG1C-SAS cache, which is used as the
auxiliary target of the score head. This requires a CUDA GPU and
`pip install pysptools imagecodecs`:

```bash
python scripts/generate_mag1c_products.py --root datasets/starcop --csv train.csv
```

Optional cube cache for faster training:

```bash
python scripts/build_starcop_npy_cache.py --root datasets/starcop --out datasets/starcop_swir_npy
```

### EMIT

OxHyperSyntheticCH4 was released by Ruzicka and Markham for HyperspectralViTs.
Original download link:
[previtus/OxHyperSyntheticCH4](https://huggingface.co/datasets/previtus/OxHyperSyntheticCH4).

```bash
python scripts/download_emit.py
```

This downloads the dataset to `datasets/oxhyper_synthetic_ch4/`. The Hub
layout is used as-is and the MAG1C product ships with each tile, so no extra
preparation is needed. Optional cube cache for faster training:

```bash
python scripts/build_emit_npy_cache.py --root datasets/oxhyper_synthetic_ch4 --out datasets/emit_npy64
```

## Training

```bash
python train.py --config configs/flame_starcop.yaml --seed 42
python train.py --config configs/flame_emit.yaml --seed 42
```

`train.batch_size` must be divisible by the number of GPUs. Checkpoints and
TensorBoard logs are written to `logs/<uid>/seed_<seed>/`.

## Evaluation

```bash
python evaluate.py --uid flame_starcop
python evaluate.py --uid flame_emit
```

## Visualization

```bash
python visualize.py --uid flame_starcop --max-tiles 20
python visualize.py --uid flame_emit --all-tiles --max-tiles 40
```

## License

Apache-2.0. The datasets and MAG1C are distributed under their own licenses.

## Citation

```bibtex
@article{heo2026flame,
  title={FLAME: Physics-Guided Neural Operators for Onboard Satellite Methane Detection in Hyperspectral Imagery},
  author={Heo, Junhyuk and Park, Junhwan and Sim, Sancheol and Choi, Beomkyu and Cho, Woojin},
  journal={arXiv preprint arXiv:2606.01577},
  year={2026}
}
```
