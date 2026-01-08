import torch
import torch.nn as nn

class ConvTransposeAETest(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 784, "This model is designed for latent_dim=64"

        # Pre-encoder: 3 → 8 → 16
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(48, 56, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(56),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(56, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1)
        )

        flat_dim = 64 * 7 * 7
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 56, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(56),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(56, 48, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.final = nn.Conv2d(16, image_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def encode(self, x):
        x0 = self.pre_encoder(x)  # 224x224
        x1 = self.enc1(x0)  # 112x112
        x2 = self.enc2(x1)  # 56x56
        x3 = self.enc3(x2)  # 28x28
        x4 = self.enc4(x3)  # 14x14
        z_map = self.enc5(x4)  # 7x7

        b = z_map.size(0)
        z_vec = self.fc_enc(z_map.flatten(1))  # (B, latent_dim)
        return z_vec

    def decode(self, z_vec):
        b = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(b, 64, 7, 7)

        d1 = self.dec1(z_map)  # 14x14
        d2 = self.dec2(d1)  # 28x28
        d3 = self.dec3(d2)  # 56x56
        d4 = self.dec4(d3)  # 112x112
        d5 = self.dec5(d4)  # 224x224
        return self.activation(self.final(d5))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# Required by main.py
model_class = ConvTransposeAETest
config_path = "configs/test/conv_transpose_ae_test.json"