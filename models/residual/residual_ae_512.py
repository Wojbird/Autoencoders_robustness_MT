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


class ResidualAutoencoderAE512(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 6272, "This model is designed for latent_dim=6272"

        # Pre-encoder: 3 → 32 → 64
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),    # 32x224x224
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 64x224x224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128x112x112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(128)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),    # 256x56x56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(256)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),    # 384x28x28
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(384)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(384, 448, kernel_size=3, stride=2, padding=1),    # 448x14x14
            nn.BatchNorm2d(448),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(448)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(448, 512, kernel_size=3, stride=2, padding=1), # 512x7x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(512)
        )

        flat_dim = 512 * 7 * 7  # 64*7*7=3136 for 224
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 448, kernel_size=3, stride=2, padding=1, output_padding=1),  # 448x14x14
            nn.BatchNorm2d(448),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(448)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(448, 384, kernel_size=3, stride=2, padding=1, output_padding=1),   # 384xz28x28
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(384)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1),   # 256x56x56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            ResBlock(256)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   # 128x112x112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(128)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),    # 64x224x224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64)
        )

        self.final = nn.Conv2d(64, image_channels, kernel_size=3, padding=1)    # 3x224x224
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
        z_map = self.fc_dec(z_vec).view(b, 512, 7, 7)

        d1 = self.dec1(z_map)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)
        return self.activation(self.final(d5))

model_class = ResidualAutoencoderAE512
config_path = "configs/residual/residual_ae_512.json"