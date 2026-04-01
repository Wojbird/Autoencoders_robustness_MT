import torch
import torch.nn as nn


class UNetBlock(nn.Module):
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


class ImageDiscriminator(nn.Module):
    def __init__(self, image_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdversarialUNetAEBase(nn.Module):
    discriminator_class = ImageDiscriminator

    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_dim = int(config["latent_dim"])

        enc_channels = list(config["enc_channels"])
        bottleneck_channels = int(config["bottleneck_channels"])
        block_dropout = float(config.get("block_dropout", 0.0))

        if len(enc_channels) != 5:
            raise ValueError("config['enc_channels'] must contain exactly 5 values.")

        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        self._latent_hw = image_size // 32
        self._bottleneck_channels = bottleneck_channels

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = UNetBlock(image_channels, enc_channels[0], dropout=0.0)
        self.enc2 = UNetBlock(enc_channels[0], enc_channels[1], dropout=0.0)
        self.enc3 = UNetBlock(enc_channels[1], enc_channels[2], dropout=block_dropout)
        self.enc4 = UNetBlock(enc_channels[2], enc_channels[3], dropout=block_dropout)
        self.enc5 = UNetBlock(enc_channels[3], enc_channels[4], dropout=block_dropout)

        # Bottleneck
        self.bottleneck = UNetBlock(enc_channels[4], bottleneck_channels, dropout=block_dropout)

        flat_dim = bottleneck_channels * self._latent_hw * self._latent_hw
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(bottleneck_channels, enc_channels[4], kernel_size=2, stride=2)
        self.dec1 = UNetBlock(enc_channels[4], enc_channels[4], dropout=block_dropout)

        self.up2 = nn.ConvTranspose2d(enc_channels[4], enc_channels[3], kernel_size=2, stride=2)
        self.dec2 = UNetBlock(enc_channels[3], enc_channels[3], dropout=block_dropout)

        self.up3 = nn.ConvTranspose2d(enc_channels[3], enc_channels[2], kernel_size=2, stride=2)
        self.dec3 = UNetBlock(enc_channels[2], enc_channels[2], dropout=block_dropout)

        self.up4 = nn.ConvTranspose2d(enc_channels[2], enc_channels[1], kernel_size=2, stride=2)
        self.dec4 = UNetBlock(enc_channels[1], enc_channels[1], dropout=0.0)

        self.up5 = nn.ConvTranspose2d(enc_channels[1], enc_channels[0], kernel_size=2, stride=2)
        self.dec5 = UNetBlock(enc_channels[0], enc_channels[0], dropout=0.0)

        self.final = nn.Conv2d(enc_channels[0], image_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        z_map = self.bottleneck(self.pool(e5))
        z_vec = self.fc_enc(z_map.flatten(1))
        return z_vec

    def decode(self, z_vec: torch.Tensor) -> torch.Tensor:
        batch_size = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(
            batch_size,
            self._bottleneck_channels,
            self._latent_hw,
            self._latent_hw
        )

        d1 = self.dec1(self.up1(z_map))
        d2 = self.dec2(self.up2(d1))
        d3 = self.dec3(self.up3(d2))
        d4 = self.dec4(self.up4(d3))
        d5 = self.dec5(self.up5(d4))

        return self.activation(self.final(d5))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z