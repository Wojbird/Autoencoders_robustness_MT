import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = ResBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.res(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = ResBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.res(x)


class ResidualAEBase(nn.Module):
    """
    Fully-convolutional residual autoencoder.

    Latent representation:
        [B, latent_channels, image_size / 32, image_size / 32]

    For image_size=224:
        [B, latent_channels, 7, 7]

    This version intentionally removes fc_enc/fc_dec.
    """

    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_channels = int(config.get("latent_channels", config.get("latent_dim")))

        stem_channels = list(config["stem_channels"])
        encoder_channels = list(config["encoder_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(stem_channels) != 2:
            raise ValueError("config['stem_channels'] must contain exactly 2 values.")
        if len(encoder_channels) != 5:
            raise ValueError("config['encoder_channels'] must contain exactly 5 values.")
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        s1, s2 = map(int, stem_channels)
        c1, c2, c3, c4, c5 = map(int, encoder_channels)

        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_hw = image_size // 32
        self.latent_size = latent_channels * self.latent_hw * self.latent_hw

        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, s1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(s1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s1, s2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(s2),
            nn.ReLU(inplace=True),
        )

        self.enc1 = DownBlock(s2, c1, dropout=0.0)
        self.enc2 = DownBlock(c1, c2, dropout=0.0)
        self.enc3 = DownBlock(c2, c3, dropout=dropout)
        self.enc4 = DownBlock(c3, c4, dropout=dropout)
        self.enc5 = DownBlock(c4, c5, dropout=dropout)

        self.to_latent = nn.Sequential(
            nn.Conv2d(c5, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, c5, kernel_size=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        self.dec1 = UpBlock(c5, c4, dropout=dropout)
        self.dec2 = UpBlock(c4, c3, dropout=dropout)
        self.dec3 = UpBlock(c3, c2, dropout=0.0)
        self.dec4 = UpBlock(c2, c1, dropout=0.0)
        self.dec5 = UpBlock(c1, s2, dropout=0.0)

        self.final = nn.Sequential(
            nn.Conv2d(s2, s1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(s1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s1, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        z = self.to_latent(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)