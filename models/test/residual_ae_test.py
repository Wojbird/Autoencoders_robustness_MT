import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResidualAETest(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 64, "This model is designed for latent_dim=64"

        # Pre-encoder: 3 → 8 → 16
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(24)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(32)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(48)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(48, 56, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(56),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(56)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(56, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(latent_dim)
        )

        # Decoder blocks with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 56, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(56),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(56)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(56 + 56, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(48)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(48 + 48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(32)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(24)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(24 + 24, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(16)
        )

        self.final = nn.Conv2d(16, image_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x):
        x0 = self.pre_encoder(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        self._skips = [x1, x2, x3, x4]
        return x5

    def decode(self, z):
        x4, x3, x2, x1 = self._skips
        d1 = self.dec1(z)
        d2 = self.dec2(torch.cat([d1, x4], dim=1))
        d3 = self.dec3(torch.cat([d2, x3], dim=1))
        d4 = self.dec4(torch.cat([d3, x2], dim=1))
        d5 = self.dec5(torch.cat([d4, x1], dim=1))
        return self.activation(self.final(d5))


# Required by main.py
model_class = ResidualAETest
config_path = "configs/test/residual_ae_test.json"