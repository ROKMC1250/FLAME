"""FLAME (Fourier Learned Absorption Matched Estimator): physics-guided
neural operator for methane plume segmentation.

Architecture:
    SWIR -> FNO/U-FNO backbone (channel-SE + SiLU) -> [bg_head, sw_head]
         -> parameter-free physics score (aux supervised)
         -> cat[feat, score_norm, rgb_norm] -> conv seg head -> logits

Input:  (B, C, H, W) SWIR cube + (B, 3, H, W) RGB
Output: (B, 1, H, W) segmentation logits, (B, 1, H, W) normalized score

The CH4 absorption spectrum, the band centers, and the background-mean
initialisation statistics are loaded from a per-dataset resource directory
(``resources/starcop`` for AVIRIS-NG/STARCOP, ``resources/emit`` for
EMIT/OxHyperSyntheticCH4); see ``configs/flame_*.yaml``.

``forward`` optionally accepts an externally computed matched-filter product
``mag`` (mag1c). When given, the segmentation head receives ``mag`` (divided by
``score_divisor``, clamped to [0, 2]) in place of the physics score — the
regime used on EMIT, where mag1c is available at train and inference time.
Without ``mag`` the model is fully self-contained (STARCOP regime).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path):
    """Resolve a resource path relative to the repository root."""
    if path is None:
        return None
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    return path


# ============================================================
# Spectral Convolution 2D
# ============================================================

class SpectralConv2d(nn.Module):
    """FFT -> learned complex linear on low-freq modes -> IFFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x):
        batch, C, H, W = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfft2(x_fp32)
            w1 = torch.view_as_complex(self.weights1)
            w2 = torch.view_as_complex(self.weights2)
            out_ft = torch.zeros(batch, self.out_channels, H, W // 2 + 1,
                                 dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                'bixy,ioxy->boxy', x_ft[:, :, :self.modes1, :self.modes2], w1)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                'bixy,ioxy->boxy', x_ft[:, :, -self.modes1:, :self.modes2], w2)
            return torch.fft.irfft2(out_ft, s=(H, W))


# ============================================================
# Mini U-Net 2D (local spatial path in U-FNO layer)
# ============================================================

class UNet2d(nn.Module):
    """Lightweight 3-level U-Net operating in feature space.
    Captures local spatial patterns to complement FNO's global spectral view."""

    def __init__(self, channels, kernel_size=3, dropout_rate=0.0):
        super().__init__()
        # Encoder
        self.enc1 = self._conv(channels, channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.enc2 = nn.Sequential(
            self._conv(channels, channels, kernel_size, stride=2, dropout_rate=dropout_rate),
            self._conv(channels, channels, kernel_size, stride=1, dropout_rate=dropout_rate),
        )
        self.enc3 = nn.Sequential(
            self._conv(channels, channels, kernel_size, stride=2, dropout_rate=dropout_rate),
            self._conv(channels, channels, kernel_size, stride=1, dropout_rate=dropout_rate),
        )
        # Decoder
        self.dec3 = self._deconv(channels, channels)
        self.dec2 = self._deconv(channels * 2, channels)
        self.dec1 = self._deconv(channels * 2, channels)
        # Output (concat with input -> project back)
        self.out_conv = nn.Conv2d(channels * 2, channels, kernel_size=kernel_size,
                                  padding=kernel_size // 2)

    def forward(self, x):
        # Pad to multiple of 8 for clean downsampling
        _, _, H, W = x.shape
        pH = (8 - H % 8) % 8
        pW = (8 - W % 8) % 8
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, pW, 0, pH))

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d3 = self.dec3(e3)
        d3 = self._match_size(d3, e2)
        d2 = self.dec2(torch.cat([e2, d3], dim=1))
        d2 = self._match_size(d2, e1)
        d1 = self.dec1(torch.cat([e1, d2], dim=1))
        d1 = self._match_size(d1, x)
        out = self.out_conv(torch.cat([x, d1], dim=1))

        # Remove padding
        if pH > 0 or pW > 0:
            out = out[:, :, :H, :W]
        return out

    @staticmethod
    def _match_size(x, target):
        """Crop or pad x to match target's spatial dims."""
        _, _, tH, tW = target.shape
        _, _, xH, xW = x.shape
        if xH > tH or xW > tW:
            x = x[:, :, :tH, :tW]
        if xH < tH or xW < tW:
            x = F.pad(x, (0, tW - xW, 0, tH - xH))
        return x

    @staticmethod
    def _conv(in_ch, out_ch, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    @staticmethod
    def _deconv(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )


# ============================================================
# Channel Squeeze-and-Excitation
# ============================================================

class ChannelSE(nn.Module):
    """SE-Net style channel attention on the width-dim feature.

    GAP -> FC(channels->mid) -> ReLU -> FC(mid->channels) -> Sigmoid -> scale.
    The inner ReLU matches the original SE-Net bottleneck; SiLU is used only on
    the main feed-forward path.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gap = self.pool(x).squeeze(-1).squeeze(-1)
        s = self.sigmoid(self.fc2(self.act(self.fc1(gap))))
        return x * s.unsqueeze(-1).unsqueeze(-1)


# ============================================================
# FNO / U-FNO layers
# ============================================================

class FNOLayer(nn.Module):
    """SpectralConv + 1x1 skip -> channel SE -> SiLU."""

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.skip = nn.Conv2d(width, width, 1)
        self.se = ChannelSE(width)

    def forward(self, x):
        out = self.spectral_conv(x) + self.skip(x)
        out = self.se(out)
        return F.silu(out)


class UFNOLayer(nn.Module):
    """SpectralConv + 1x1 skip + U-Net path -> channel SE -> SiLU."""

    def __init__(self, width, modes1, modes2, kernel_size=3, dropout_rate=0.0):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.skip = nn.Conv2d(width, width, 1)
        self.unet = UNet2d(width, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.se = ChannelSE(width)

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.skip(x)
        x3 = self.unet(x)
        out = x1 + x2 + x3
        out = self.se(out)
        return F.silu(out)


# ============================================================
# FLAME model
# ============================================================

class FLAME(nn.Module):
    """FLAME segmentation model.

    Args:
        in_channels: number of SWIR bands (72 for STARCOP, 64 for EMIT).
        width: hidden channel dimension of the FNO/U-FNO layers.
        modes1, modes2: number of retained Fourier modes per spatial dimension.
        n_fno_layers: number of pure FNO layers.
        n_ufno_layers: number of U-FNO layers (FNO + U-Net path).
        dropout_rate: dropout inside the U-Net path.
        seg_channels: hidden channel widths of the segmentation head.
        seg_kernel_size: kernel size of the segmentation head convs.
        score_divisor: normalisation constant for the physics score (and for
            ``mag`` when it is fed to the segmentation head).
        rgb_divisor: normalisation constant for RGB.
        norm_type: 'batch' | 'group' | 'none' — normalisation in the seg head.
        score_clamp: (min, max) clamp of the normalised physics score.
        spectrum_path: .npy CH4 absorption spectrum. Either already sliced to
            ``in_channels`` entries, or full-resolution together with
            ``centers_path`` + ``wv_range`` for slicing at load time.
        centers_path: .npy band centers matching ``spectrum_path`` (optional).
        baseline_stats_path: .pt with key ``mu_mean`` — mean background
            spectrum used to initialise ``bg_head.bias`` (optional).
        wv_range: (lo, hi) nm window used to slice the spectrum when
            ``centers_path`` is given.
    """

    def __init__(self, in_channels=72, width=32, modes1=16, modes2=16,
                 n_fno_layers=3, n_ufno_layers=3, dropout_rate=0.0,
                 seg_channels=None, seg_kernel_size=3,
                 score_divisor=1750.0, rgb_divisor=60.0,
                 norm_type='batch', score_clamp=(0.0, 2.0),
                 spectrum_path='resources/starcop/ch4_spectrum.npy',
                 centers_path=None, baseline_stats_path=None,
                 wv_range=(2122, 2488)):
        super().__init__()
        if seg_channels is None:
            seg_channels = [64, 32]

        self.in_channels = in_channels
        self.score_divisor = score_divisor
        self.rgb_divisor = rgb_divisor
        self.score_clamp_min = float(score_clamp[0])
        self.score_clamp_max = float(score_clamp[1])

        # CH4 absorption spectrum (fixed, not learned)
        ch4_spectrum = self._load_ch4_spectrum(
            _resolve(spectrum_path), _resolve(centers_path), wv_range, in_channels)
        self.register_buffer('ch4_spectrum', ch4_spectrum)  # (in_channels,)

        # --- Shared Backbone ---
        self.lift = nn.Conv2d(in_channels, width, 1)

        self.fno_layers = nn.ModuleList([
            FNOLayer(width, modes1, modes2) for _ in range(n_fno_layers)
        ])
        self.ufno_layers = nn.ModuleList([
            UFNOLayer(width, modes1, modes2, dropout_rate=dropout_rate)
            for _ in range(n_ufno_layers)
        ])

        # --- Head 1: Background Estimator ---
        self.bg_head = nn.Conv2d(width, in_channels, 1)

        # --- Head 2: Spectral Weight ---
        self.sw_head = nn.Conv2d(width, in_channels, 1)

        nn.init.xavier_uniform_(self.bg_head.weight, gain=0.1)
        mu_mean = self._load_baseline_mu(_resolve(baseline_stats_path), in_channels)
        if mu_mean is not None:
            self.bg_head.bias.data.copy_(mu_mean)
        else:
            nn.init.zeros_(self.bg_head.bias)
        nn.init.xavier_uniform_(self.sw_head.weight, gain=0.1)
        nn.init.constant_(self.sw_head.bias, 10.0)

        # --- Segmentation Head ---
        seg_in_ch = width + 1 + 3  # backbone(width) + score(1) + rgb(3)
        layers = []
        ch = seg_in_ch
        for ch_out in seg_channels:
            layers.append(nn.Conv2d(ch, ch_out, seg_kernel_size,
                                    padding=seg_kernel_size // 2,
                                    bias=(norm_type == 'none')))
            if norm_type == 'batch':
                layers.append(nn.BatchNorm2d(ch_out))
            elif norm_type == 'group':
                layers.append(nn.GroupNorm(min(8, ch_out), ch_out))
            elif norm_type != 'none':
                raise ValueError(f"norm_type must be 'batch', 'group' or 'none', got {norm_type!r}")
            layers.append(nn.SiLU(inplace=True))
            ch = ch_out
        layers.append(nn.Conv2d(ch, 1, 1))
        self.seg_head = nn.Sequential(*layers)

    @staticmethod
    def _load_ch4_spectrum(spectrum_path, centers_path, wv_range, in_bands):
        spectrum = np.load(spectrum_path)
        if len(spectrum) != in_bands:
            if centers_path is None:
                raise ValueError(
                    f'spectrum has {len(spectrum)} entries but in_channels={in_bands}; '
                    f'provide centers_path + wv_range for slicing')
            centers = np.load(centers_path)
            mask = (centers >= wv_range[0]) & (centers <= wv_range[1])
            spectrum = spectrum[mask]
        if len(spectrum) != in_bands:
            raise ValueError(
                f'expected {in_bands} spectrum entries, got {len(spectrum)} '
                f'after slicing to wv_range={wv_range}')
        return torch.tensor(spectrum, dtype=torch.float32)

    @staticmethod
    def _load_baseline_mu(stats_path, in_bands):
        """Load pre-computed mean background spectrum for bg_head init."""
        if stats_path is None or not os.path.exists(stats_path):
            return None
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        mu = stats.get('mu_mean')
        if mu is not None and len(mu) == in_bands:
            return mu.float()
        return None

    def forward(self, swir, rgb, mag=None):
        """
        Args:
            swir: (B, C, H, W) SWIR hyperspectral cube.
            rgb:  (B, 3, H, W) RGB bands.
            mag:  optional (B, H, W) or (B, 1, H, W) matched-filter product;
                  when given, it replaces the physics score at the seg head.

        Returns:
            logits:     (B, 1, H, W) segmentation logits.
            score_norm: (B, 1, H, W) normalised physics score.
        """
        # --- Shared Backbone ---
        feat = self.lift(swir)
        for layer in self.fno_layers:
            feat = layer(feat)
        for layer in self.ufno_layers:
            feat = layer(feat)

        # --- Heads + parameter-free score ---
        b = self.bg_head(feat)                                 # (B, C, H, W)
        w = self.sw_head(feat)                                 # (B, C, H, W)
        residual = swir - b
        s = self.ch4_spectrum[None, :, None, None]             # (1, C, 1, 1)
        score_raw = (residual * w * s).sum(dim=1, keepdim=True)

        score_norm = torch.clamp(score_raw / self.score_divisor,
                                 self.score_clamp_min, self.score_clamp_max)
        rgb_norm = torch.clamp(rgb / self.rgb_divisor, 0, 2)

        if mag is not None:
            mag_in = mag.unsqueeze(1) if mag.dim() == 3 else mag
            seg_score = torch.clamp(mag_in / self.score_divisor, 0.0, 2.0)
        else:
            seg_score = score_norm

        # --- Segmentation Head ---
        seg_in = torch.cat([feat, seg_score, rgb_norm], dim=1)  # (B, width+4, H, W)
        logits = self.seg_head(seg_in)

        return logits, score_norm


def build_model(model_cfg):
    """Build a FLAME model from the ``model:`` section of a config.

    Keys that configure the training/eval pipeline rather than the module
    (``use_mag_in_seg``) are ignored here.
    """
    kwargs = dict(model_cfg)
    kwargs.pop('use_mag_in_seg', None)
    return FLAME(**kwargs)
