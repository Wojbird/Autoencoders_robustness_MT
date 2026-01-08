import torch
import torch.nn as nn

class ConvTransposeAE512(nn.Module):
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
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x112x112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256x56x56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),  # 384x28x28
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(384, 448, kernel_size=3, stride=2, padding=1),  # 448x14x14
            nn.BatchNorm2d(448),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(448, 512, kernel_size=3, stride=2, padding=1),  # 512x7x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )

        flat_dim = 512 * 7 * 7
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 448, kernel_size=3, stride=2, padding=1, output_padding=1),  # 448x14x14
            nn.BatchNorm2d(448),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(448, 384, kernel_size=3, stride=2, padding=1, output_padding=1),  # 384xz28x28
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x56x56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x112x112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x224x224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.final = nn.Conv2d(64, image_channels, kernel_size=3, padding=1)    # 3x224x224
        self.activation = nn.Sigmoid()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x):
        x0 = self.pre_encoder(x)  # 224x224
        x1 = self.enc1(x0)  # 112x112
        x2 = self.enc2(x1)  # 56x56
        x3 = self.enc3(x2)  # 28x28
        x4 = self.enc4(x3)  # 14x14
        z_map = self.enc5(x4)  # 7x7

        z_vec = self.fc_enc(z_map.flatten(1))  # (B, latent_dim)
        return z_vec

    def decode(self, z_vec):
        b = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(b, 512, 7, 7)

        d1 = self.dec1(z_map)  # 14x14
        d2 = self.dec2(d1)  # 28x28
        d3 = self.dec3(d2)  # 56x56
        d4 = self.dec4(d3)  # 112x112
        d5 = self.dec5(d4)  # 224x224
        return self.activation(self.final(d5))

model_class = ConvTransposeAE512
config_path = "configs/conv/conv_transpose_ae_512.json"