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
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAEBase(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_dim = int(config["latent_dim"])

        encoder_channels = list(config["encoder_channels"])
        bottleneck_channels = int(config["bottleneck_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(encoder_channels) != 5:
            raise ValueError("config['encoder_channels'] must contain exactly 5 values.")
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        self._latent_hw = image_size // 32
        self._bottleneck_channels = bottleneck_channels
        self._skip_cache = None

        c1, c2, c3, c4, c5 = map(int, encoder_channels)
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(image_channels, c1, dropout=0.0)
        self.enc2 = DoubleConv(c1, c2, dropout=0.0)
        self.enc3 = DoubleConv(c2, c3, dropout=dropout)
        self.enc4 = DoubleConv(c3, c4, dropout=dropout)
        self.enc5 = DoubleConv(c4, c5, dropout=dropout)

        self.bottleneck = DoubleConv(c5, bottleneck_channels, dropout=dropout)

        flat_dim = bottleneck_channels * self._latent_hw * self._latent_hw
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        self.dec1 = UpBlock(bottleneck_channels, c5, c5, dropout=dropout)
        self.dec2 = UpBlock(c5, c4, c4, dropout=dropout)
        self.dec3 = UpBlock(c4, c3, c3, dropout=dropout)
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
        z_map = self.bottleneck(self.pool(e5))

        self._skip_cache = (e1, e2, e3, e4, e5)
        z_vec = self.fc_enc(z_map.flatten(1))
        return z_vec

    def decode(self, z_vec: torch.Tensor) -> torch.Tensor:
        if self._skip_cache is None:
            raise RuntimeError("UNetAEBase.decode() called before encode(); skip cache is empty.")

        e1, e2, e3, e4, e5 = self._skip_cache

        batch_size = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(
            batch_size,
            self._bottleneck_channels,
            self._latent_hw,
            self._latent_hw,
        )

        d1 = self.dec1(z_map, e5)
        d2 = self.dec2(d1, e4)
        d3 = self.dec3(d2, e3)
        d4 = self.dec4(d3, e2)
        d5 = self.dec5(d4, e1)

        return self.activation(self.final(d5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)