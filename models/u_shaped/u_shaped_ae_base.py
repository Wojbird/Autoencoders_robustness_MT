from __future__ import annotations

import torch
import torch.nn as nn

from models.common.blocks import (
    DecoderUpBlock,
    DoubleConv,
    group_norm,
    validate_autoencoder_config,
)


class UShapedAEBase(nn.Module):
    """Symmetric U-shaped autoencoder without skip connections.

    Every reconstruction path passes through the central tensor of shape
    ``[B, latent_channels, image_size // 32, image_size // 32]``.
    """

    def __init__(self, config: dict):
        super().__init__()

        image_channels, image_size, latent_channels, channels, dropout = (
            validate_autoencoder_config(config)
        )
        c1, c2, c3, c4, c5 = channels

        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_hw = image_size // 32
        self.latent_size = latent_channels * self.latent_hw * self.latent_hw

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(image_channels, c1, dropout=0.0)
        self.enc2 = DoubleConv(c1, c2, dropout=0.0)
        self.enc3 = DoubleConv(c2, c3, dropout=0.0)
        self.enc4 = DoubleConv(c3, c4, dropout=dropout)
        self.enc5 = DoubleConv(c4, c5, dropout=dropout)
        self.bottleneck = DoubleConv(c5, c5, dropout=dropout)

        self.to_latent = nn.Sequential(
            nn.Conv2d(c5, latent_channels, kernel_size=1, bias=False),
            group_norm(latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, c5, kernel_size=1, bias=False),
            group_norm(c5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.dec1 = DecoderUpBlock(c5, c5, dropout=dropout)
        self.dec2 = DecoderUpBlock(c5, c4, dropout=dropout)
        self.dec3 = DecoderUpBlock(c4, c3, dropout=0.0)
        self.dec4 = DecoderUpBlock(c3, c2, dropout=0.0)
        self.dec5 = DecoderUpBlock(c2, c1, dropout=0.0)
        self.final = nn.Sequential(
            nn.Conv2d(c1, image_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(self.pool(x))
        x = self.enc3(self.pool(x))
        x = self.enc4(self.pool(x))
        x = self.enc5(self.pool(x))
        x = self.bottleneck(self.pool(x))
        return self.to_latent(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        return self.final(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
