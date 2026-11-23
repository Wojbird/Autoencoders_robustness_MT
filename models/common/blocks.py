from __future__ import annotations

import torch
import torch.nn as nn


def num_groups(channels: int) -> int:
    """Choose the largest small GroupNorm divisor used across the project."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups(channels), num_channels=channels)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderUpBlock(nn.Module):
    """Upsampling block without a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.conv = DoubleConv(out_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class SkipDecoderUpBlock(nn.Module):
    """Upsample, concatenate the encoder feature map, then fuse it."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.conv = DoubleConv(2 * out_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "Decoder and skip spatial shapes differ: "
                f"decoder={tuple(x.shape[-2:])}, skip={tuple(skip.shape[-2:])}."
            )
        if x.shape[0] != skip.shape[0]:
            raise ValueError("Decoder and skip batch sizes differ.")
        return self.conv(torch.cat((x, skip), dim=1))


def validate_autoencoder_config(config: dict) -> tuple[int, int, int, list[int], float]:
    image_channels = int(config["image_channels"])
    image_size = int(config.get("image_size", 224))
    latent_channels = int(config.get("latent_channels", config.get("latent_dim")))
    encoder_channels = [int(value) for value in config["encoder_channels"]]
    dropout = float(config.get("dropout", 0.2))

    if image_channels <= 0:
        raise ValueError("image_channels must be positive.")
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError("image_size must be positive and divisible by 32.")
    if latent_channels <= 0:
        raise ValueError("latent_channels must be positive.")
    if len(encoder_channels) != 5:
        raise ValueError("config['encoder_channels'] must contain exactly 5 values.")
    if any(value <= 0 for value in encoder_channels):
        raise ValueError("All encoder_channels values must be positive.")
    if any(left >= right for left, right in zip(encoder_channels, encoder_channels[1:])):
        raise ValueError("encoder_channels must be strictly increasing.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the [0, 1) interval.")

    return image_channels, image_size, latent_channels, encoder_channels, dropout
