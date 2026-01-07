import torch
import torch.nn as nn

class ConvTransposeAE128(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 128, "This model is designed for latent_dim=128"

        # Pre-encoder
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
            nn.Conv2d(64, 80, kernel_size=3, stride=2, padding=1),  # 80x112x112
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(80, 96, kernel_size=3, stride=2, padding=1),  # 96x56x56
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(96, 112, kernel_size=3, stride=2, padding=1),  # 112x28x28
            nn.BatchNorm2d(112),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(112, 120, kernel_size=3, stride=2, padding=1),  # 120x14x14
            nn.BatchNorm2d(120),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(120, latent_dim, kernel_size=3, stride=2, padding=1),  # 128x7x7
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 120, kernel_size=3, stride=2, padding=1, output_padding=1),  # 120x14x14
            nn.BatchNorm2d(120),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(120, 112, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x28x28
            nn.BatchNorm2d(112),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(112, 96, kernel_size=3, stride=2, padding=1, output_padding=1),  # 96x56x56
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(96, 80, kernel_size=3, stride=2, padding=1, output_padding=1),  # 80x112x112
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(80, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x224x224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.final = nn.Conv2d(64, image_channels, kernel_size=3, padding=1)    # 3x224x224
        self.activation = nn.Tanh()

    def forward(self, x):
        x_latent = self.encode(x)
        return self.decode(x_latent)

    def encode(self, x):
        x0 = self.pre_encoder(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        return x5

    def decode(self, z):
        d1 = self.dec1(z)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)
        return self.activation(self.final(d5))


model_class = ConvTransposeAE128
config_path = "configs/conv/conv_transpose_ae_128.json"