import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        ]
        if use_dropout:
            layers.insert(4, nn.Dropout2d(0.1))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetAETest(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 784, "This model is designed for latent_dim=784"

        self.pool = nn.MaxPool2d(2)

        # Encoder: 5 levels
        self.enc1 = UNetBlock(image_channels, 16)
        self.enc2 = UNetBlock(16, 24)
        self.enc3 = UNetBlock(24, 32, use_dropout=True)
        self.enc4 = UNetBlock(32, 48, use_dropout=True)
        self.enc5 = UNetBlock(48, 56, use_dropout=True)
        self.bottleneck = UNetBlock(56, 64, use_dropout=True)

        flat_dim = 64 * 7 * 7  # 64*7*7=3136 for 224
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder: 5 levels
        self.up1 = nn.ConvTranspose2d(64, 56, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(56, 56, use_dropout=True)

        self.up2 = nn.ConvTranspose2d(56, 48, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(48, 48, use_dropout=True)

        self.up3 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(32, 32, use_dropout=True)

        self.up4 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(24, 24)

        self.up5 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(16, 16)

        self.final = nn.Conv2d(16, image_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        z_map = self.bottleneck(self.pool(e5))

        z_vec = self.fc_enc(z_map.flatten(1))  # (B, latent_dim)
        return z_vec

    def decode(self, z_vec):
        b = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(b, 64, 7, 7)

        d1 = self.up1(z_map)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = self.dec4(d4)

        d5 = self.up5(d4)
        d5 = self.dec5(d5)
        return self.activation(self.final(d5))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


# Required by main.py
model_class = UNetAETest
config_path = "configs/test/unet_ae_test.json"