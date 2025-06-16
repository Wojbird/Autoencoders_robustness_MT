import torch
import torch.nn as nn

class ConvTransposeAE256(nn.Module):
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

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(192, 224, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(224),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(224, latent_dim, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )

        # Decoder (skip connections via concat)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + latent_dim, 224, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.BatchNorm2d(224),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(224 + 224, 192, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(192 + 192, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.final = nn.Conv2d(64, image_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_latent = self.encode(x)
        return self.decode(x_latent)

    def encode(self, x):
        x0 = self.pre_encoder(x)  # 224x224
        x1 = self.enc1(x0)        # 112x112
        x2 = self.enc2(x1)        # 56x56
        x3 = self.enc3(x2)        # 28x28
        x4 = self.enc4(x3)        # 14x14
        x5 = self.enc5(x4)        # 7x7
        # Store skip connections
        self._skips = [x1, x2, x3, x4]
        return x5

    def decode(self, z):
        x4, x3, x2, x1 = self._skips  # reverse order
        d1 = self.dec1(z)                          # 14x14
        d2 = self.dec2(torch.cat([d1, x4], dim=1)) # 28x28
        d3 = self.dec3(torch.cat([d2, x3], dim=1)) # 56x56
        d4 = self.dec4(torch.cat([d3, x2], dim=1)) # 112x112
        d5 = self.dec5(torch.cat([d4, x1], dim=1)) # 224x224
        return self.activation(self.final(d5))


# Required by main.py
model_class = ConvTransposeAE256
config_path = "configs/conv/conv_transpose_ae_256.json"