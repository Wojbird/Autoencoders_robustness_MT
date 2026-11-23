from __future__ import annotations

import torch
import torch.nn as nn

from models.common.blocks import (
    DoubleConv,
    SkipDecoderUpBlock,
    group_norm,
    validate_autoencoder_config,
)


class UNetAEBase(nn.Module):
    """True U-Net autoencoder with concatenating encoder-to-decoder skips.

    ``encode`` returns only the central representation for compatibility with
    latent-space evaluation. ``forward_with_latent_transform`` is the canonical
    path for perturbing that representation while keeping skips unchanged.
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

        self.dec1 = SkipDecoderUpBlock(c5, c5, dropout=dropout)
        self.dec2 = SkipDecoderUpBlock(c5, c4, dropout=dropout)
        self.dec3 = SkipDecoderUpBlock(c4, c3, dropout=0.0)
        self.dec4 = SkipDecoderUpBlock(c3, c2, dropout=0.0)
        self.dec5 = SkipDecoderUpBlock(c2, c1, dropout=0.0)
        self.final = nn.Sequential(
            nn.Conv2d(c1, image_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _encode_with_skips(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        center = self.bottleneck(self.pool(e5))
        return self.to_latent(center), (e1, e2, e3, e4, e5)

    def _decode_with_skips(
        self,
        z: torch.Tensor,
        skips: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(skips) != 5:
            raise ValueError(f"Expected 5 skip tensors, received {len(skips)}.")
        e1, e2, e3, e4, e5 = skips
        x = self.from_latent(z)
        x = self.dec1(x, e5)
        x = self.dec2(x, e4)
        x = self.dec3(x, e3)
        x = self.dec4(x, e2)
        x = self.dec5(x, e1)
        return self.final(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self._encode_with_skips(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "UNetAEBase.decode(z) is undefined without encoder skip tensors. "
            "Use forward(x) or forward_with_latent_transform(x, transform)."
        )

    def forward_with_latent_transform(self, x: torch.Tensor, transform) -> torch.Tensor:
        z, skips = self._encode_with_skips(x)
        transformed = transform(z)
        if not isinstance(transformed, torch.Tensor):
            raise TypeError("The latent transform must return a torch.Tensor.")
        if transformed.shape != z.shape:
            raise ValueError(
                "The latent transform must preserve shape: "
                f"before={tuple(z.shape)}, after={tuple(transformed.shape)}."
            )
        return self._decode_with_skips(transformed, skips)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_latent_transform(x, lambda z: z)
