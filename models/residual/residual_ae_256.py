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
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky_relu(x + self.block(x))


class ResidualAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 256, "This model is designed for latent_dim=256"

        # Pre-encoder: 3 → 32 → 64
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Encoder: 224 → 112 → 56 → 28 → 14 → 7
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(96),

            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(128),

            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(192),

            nn.Conv2d(192, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(latent_dim)
        )

        # Decoder: 7 → 14 → 28 → 56 → 112 → 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(192),

            nn.ConvTranspose2d(192, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(128),

            nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(96),

            nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64),

            nn.ConvTranspose2d(64, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

    def encode(self, x):
        x = self.pre_encoder(x)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# Required by main.py
model_class = ResidualAutoencoder
config_path = "configs/residual_ae_256.json"