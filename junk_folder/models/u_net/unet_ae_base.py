from collections.abc import Callable

import torch
import torch.nn as nn


def _num_groups(channels: int) -> int:
    for g in (8, 4, 2):
        if channels % g == 0:
            return g
    return 1


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=_num_groups(channels), num_channels=channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(
                "Spatial mismatch in UNetUpBlock: "
                f"upsampled={tuple(x.shape)}, "
                f"skip={tuple(skip.shape)}"
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAEBase(nn.Module):
    """
    Autoenkoder U-Net ze skip connections.

    Dla wejścia 224 × 224 centralny latent ma kształt:
        [B, latent_channels, 7, 7]

    Skip connections:
        enc1: 224 × 224
        enc2: 112 × 112
        enc3:  56 × 56
        enc4:  28 × 28
        enc5:  14 × 14
    """

    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_channels = int(
            config.get("latent_channels", config.get("latent_dim"))
        )
        encoder_channels = list(config["encoder_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(encoder_channels) != 5:
            raise ValueError(
                "config['encoder_channels'] must contain exactly 5 values."
            )

        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        c1, c2, c3, c4, c5 = map(int, encoder_channels)

        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_hw = image_size // 32
        self.latent_size = latent_channels * self.latent_hw**2

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = DoubleConv(image_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4, dropout=dropout)
        self.enc5 = DoubleConv(c4, c5, dropout=dropout)

        self.bottleneck = DoubleConv(c5, c5, dropout=dropout)

        self.to_latent = nn.Sequential(
            nn.Conv2d(c5, latent_channels, kernel_size=1, bias=False),
            _norm(latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, c5, kernel_size=1, bias=False),
            _norm(c5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.dec1 = UNetUpBlock(c5, c5, c5, dropout=dropout)
        self.dec2 = UNetUpBlock(c5, c4, c4, dropout=dropout)
        self.dec3 = UNetUpBlock(c4, c3, c3)
        self.dec4 = UNetUpBlock(c3, c2, c2)
        self.dec5 = UNetUpBlock(c2, c1, c1)

        self.final = nn.Sequential(
            nn.Conv2d(c1, image_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode_with_skips(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        s5 = self.enc5(self.pool(s4))

        bottleneck = self.bottleneck(self.pool(s5))
        z = self.to_latent(bottleneck)

        return z, (s1, s2, s3, s4, s5)

    def decode_with_skips(
        self,
        z: torch.Tensor,
        skips: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(skips) != 5:
            raise ValueError(
                f"Expected 5 skip tensors, received {len(skips)}."
            )

        s1, s2, s3, s4, s5 = skips

        x = self.from_latent(z)
        x = self.dec1(x, s5)
        x = self.dec2(x, s4)
        x = self.dec3(x, s3)
        x = self.dec4(x, s2)
        x = self.dec5(x, s1)

        return self.final(x)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode_with_skips(x)
        return z

    def forward_with_latent_transform(
        self,
        x: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        z, skips = self.encode_with_skips(x)
        z = transform(z)
        return self.decode_with_skips(z, skips)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, skips = self.encode_with_skips(x)
        return self.decode_with_skips(z, skips)