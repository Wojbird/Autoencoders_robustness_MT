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


class UNetLikeAETest(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_dim = int(config["latent_dim"])
        dropout = float(config.get("dropout", 0.1))

        enc_channels = list(config.get("enc_channels", [16, 24, 32, 48, 56]))
        bottleneck_channels = int(config.get("bottleneck_channels", 64))

        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")
        if len(enc_channels) != 5:
            raise ValueError("enc_channels must contain exactly 5 values.")

        self._latent_hw = image_size // 32
        self._bottleneck_channels = bottleneck_channels
        self.pool = nn.MaxPool2d(2)

        self.enc1 = UNetBlock(image_channels, enc_channels[0], dropout=0.0)
        self.enc2 = UNetBlock(enc_channels[0], enc_channels[1], dropout=0.0)
        self.enc3 = UNetBlock(enc_channels[1], enc_channels[2], dropout=dropout)
        self.enc4 = UNetBlock(enc_channels[2], enc_channels[3], dropout=dropout)
        self.enc5 = UNetBlock(enc_channels[3], enc_channels[4], dropout=dropout)

        self.bottleneck = UNetBlock(enc_channels[4], bottleneck_channels, dropout=dropout)

        flat_dim = bottleneck_channels * self._latent_hw * self._latent_hw
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        self.up1 = nn.ConvTranspose2d(bottleneck_channels, enc_channels[4], kernel_size=2, stride=2)
        self.dec1 = UNetBlock(enc_channels[4], enc_channels[4], dropout=dropout)

        self.up2 = nn.ConvTranspose2d(enc_channels[4], enc_channels[3], kernel_size=2, stride=2)
        self.dec2 = UNetBlock(enc_channels[3], enc_channels[3], dropout=dropout)

        self.up3 = nn.ConvTranspose2d(enc_channels[3], enc_channels[2], kernel_size=2, stride=2)
        self.dec3 = UNetBlock(enc_channels[2], enc_channels[2], dropout=dropout)

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
        return self.fc_enc(z_map.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        z_map = self.fc_dec(z).view(b, self._bottleneck_channels, self._latent_hw, self._latent_hw)
        d1 = self.dec1(self.up1(z_map))
        d2 = self.dec2(self.up2(d1))
        d3 = self.dec3(self.up3(d2))
        d4 = self.dec4(self.up4(d3))
        d5 = self.dec5(self.up5(d4))
        return self.activation(self.final(d5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


model_class = UNetLikeAETest
config_path = "configs/test/unet_like_ae_test.json"