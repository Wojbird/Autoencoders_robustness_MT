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


class ResidualAE64(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 784, "This model is designed for latent_dim=784"

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
            nn.Conv2d(56, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(64)
        )

        flat_dim = 64 * 7 * 7  # 64*7*7=3136 for 224
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder blocks with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 56, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(56),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(56)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(56, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(48)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock(32)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(24)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
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
        z_map = self.enc5(x4)

        z_vec = self.fc_enc(z_map.flatten(1))  # (B, latent_dim)
        return z_vec

    def decode(self, z_vec):
        b = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(b, 64, 7, 7)

        d1 = self.dec1(z_map)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)
        return self.activation(self.final(d5))


# Required by main.py
model_class = ResidualAE64
config_path = "configs/residual/residual_ae_64.json"