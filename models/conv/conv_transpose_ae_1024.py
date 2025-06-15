import torch
import torch.nn as nn

class ConvTransposeAE1024(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 1024, "This model is designed for latent_dim=1024"

        # Pre-encoder: 3 → 32 → 64
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder: 224 → 112 → 56 → 28 → 14 → 7
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(768, latent_dim, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder: 7 → 14 → 28 → 56 → 112 → 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 768, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(512, 384, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(256, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pre_encoder(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Required by main.py
model_class = ConvTransposeAE1024
config_path = "configs/conv/conv_transpose_ae_1024.json"