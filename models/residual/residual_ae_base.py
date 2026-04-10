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
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = ResBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


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
        return self.res(self.up(x))


class ResidualAEBase(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_dim = int(config["latent_dim"])

        stem_channels = list(config["stem_channels"])
        encoder_channels = list(config["encoder_channels"])
        bottleneck_channels = int(config["bottleneck_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(stem_channels) != 2:
            raise ValueError("config['stem_channels'] must contain exactly 2 values.")
        if len(encoder_channels) != 5:
            raise ValueError("config['encoder_channels'] must contain exactly 5 values.")
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        self._latent_hw = image_size // 32
        self._bottleneck_channels = bottleneck_channels

        s1, s2 = map(int, stem_channels)
        c1, c2, c3, c4, c5 = map(int, encoder_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, s1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(s1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s1, s2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(s2),
            nn.ReLU(inplace=True),
        )

        self.enc1 = DownBlock(s2, c1, dropout=0.0)
        self.enc2 = DownBlock(c1, c2, dropout=dropout)
        self.enc3 = DownBlock(c2, c3, dropout=dropout)
        self.enc4 = DownBlock(c3, c4, dropout=dropout)
        self.enc5 = DownBlock(c4, c5, dropout=dropout)

        self.to_bottleneck = nn.Sequential(
            nn.Conv2d(c5, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        flat_dim = bottleneck_channels * self._latent_hw * self._latent_hw
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        self.dec1 = UpBlock(bottleneck_channels, c5, dropout=dropout)
        self.dec2 = UpBlock(c5, c4, dropout=dropout)
        self.dec3 = UpBlock(c4, c3, dropout=dropout)
        self.dec4 = UpBlock(c3, c2, dropout=dropout)
        self.dec5 = UpBlock(c2, s2, dropout=0.0)

        self.final = nn.Sequential(
            nn.Conv2d(s2, s1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(s1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s1, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _encode_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.to_bottleneck(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_map = self._encode_map(x)
        return self.fc_enc(z_map.flatten(1))

    def decode(self, z_vec: torch.Tensor) -> torch.Tensor:
        batch_size = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(
            batch_size,
            self._bottleneck_channels,
            self._latent_hw,
            self._latent_hw,
        )

        x = self.dec1(z_map)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)