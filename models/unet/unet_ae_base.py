import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
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
        )
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAEBase(nn.Module):
    """
    U-Net-like autoencoder with spatial latent representation.

    Latent representation:
        [B, latent_channels, image_size / 32, image_size / 32]

    For image_size=224:
        [B, latent_channels, 7, 7]

    Important:
        This model uses skip connections. It is excellent for reconstruction,
        but latent-only robustness is partially bypassed by skips.
    """

    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_channels = int(config.get("latent_channels", config.get("latent_dim")))

        encoder_channels = list(config["encoder_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(encoder_channels) != 5:
            raise ValueError("config['encoder_channels'] must contain exactly 5 values.")
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        c1, c2, c3, c4, c5 = map(int, encoder_channels)

        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_hw = image_size // 32
        self.latent_size = latent_channels * self.latent_hw * self.latent_hw
        self._skip_cache = None

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(image_channels, c1, dropout=0.0)
        self.enc2 = DoubleConv(c1, c2, dropout=0.0)
        self.enc3 = DoubleConv(c2, c3, dropout=0.0)
        self.enc4 = DoubleConv(c3, c4, dropout=dropout)
        self.enc5 = DoubleConv(c4, c5, dropout=dropout)

        self.bottleneck = DoubleConv(c5, c5, dropout=dropout)

        self.to_latent = nn.Sequential(
            nn.Conv2d(c5, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, c5, kernel_size=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.dec1 = UpBlock(c5, c5, c5, dropout=dropout)
        self.dec2 = UpBlock(c5, c4, c4, dropout=dropout)
        self.dec3 = UpBlock(c4, c3, c3, dropout=0.0)
        self.dec4 = UpBlock(c3, c2, c2, dropout=0.0)
        self.dec5 = UpBlock(c2, c1, c1, dropout=0.0)

        self.final = nn.Conv2d(c1, image_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        b = self.bottleneck(self.pool(e5))
        z = self.to_latent(b)

        self._skip_cache = (e1, e2, e3, e4, e5)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self._skip_cache is None:
            raise RuntimeError(
                "UNetAEBase.decode() called before encode(); skip cache is empty."
            )

        e1, e2, e3, e4, e5 = self._skip_cache

        x = self.from_latent(z)
        x = self.dec1(x, e5)
        x = self.dec2(x, e4)
        x = self.dec3(x, e3)
        x = self.dec4(x, e2)
        x = self.dec5(x, e1)

        return self.activation(self.final(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)