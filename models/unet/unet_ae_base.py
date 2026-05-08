import torch
import torch.nn as nn


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
            out_channels,
            out_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class UNetAEBase(nn.Module):
    """
    U-Net-like autoencoder without skip connections.

    This model intentionally removes encoder-decoder skip connections so that
    all reconstruction information must pass through the latent representation.

    Latent representation:
        [B, latent_channels, image_size / 32, image_size / 32]

    For image_size=224:
        [B, latent_channels, 7, 7]
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

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = DoubleConv(image_channels, c1, dropout=0.0)      # 224 x 224
        self.enc2 = DoubleConv(c1, c2, dropout=0.0)                  # 112 x 112
        self.enc3 = DoubleConv(c2, c3, dropout=0.0)                  # 56 x 56
        self.enc4 = DoubleConv(c3, c4, dropout=dropout)              # 28 x 28
        self.enc5 = DoubleConv(c4, c5, dropout=dropout)              # 14 x 14

        self.bottleneck = DoubleConv(c5, c5, dropout=dropout)        # 7 x 7

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

        self.dec1 = UpBlock(c5, c5, dropout=dropout)                 # 14 x 14
        self.dec2 = UpBlock(c5, c4, dropout=dropout)                 # 28 x 28
        self.dec3 = UpBlock(c4, c3, dropout=0.0)                     # 56 x 56
        self.dec4 = UpBlock(c3, c2, dropout=0.0)                     # 112 x 112
        self.dec5 = UpBlock(c2, c1, dropout=0.0)                     # 224 x 224

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
        z = self.to_latent(x)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        return self.final(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)