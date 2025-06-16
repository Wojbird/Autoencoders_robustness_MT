import torch
import torch.nn as nn

class ConvTransposeAETest(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 64, "This model is designed for latent_dim=64"

        # Pre-encoder: 3 → 8 → 16
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder: 224 → 112 → 56 → 28 → 14 → 7
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder: 7 → 14 → 28 → 56 → 112 → 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(32, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(24, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
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
model_class = ConvTransposeAETest
config_path = "configs/test/conv_transpose_ae_test.json"