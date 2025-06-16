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

class AdversarialUNetAETest(nn.Module):
    discriminator_class = None  # placeholder, nadpiszemy niżej

    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 64, "This model is designed for latent_dim=64"

        self.pool = nn.MaxPool2d(2)

        # Encoder: 5 levels
        self.enc1 = UNetBlock(image_channels, 16)
        self.enc2 = UNetBlock(16, 24)
        self.enc3 = UNetBlock(24, 32, use_dropout=True)
        self.enc4 = UNetBlock(32, 48, use_dropout=True)
        self.enc5 = UNetBlock(48, 56, use_dropout=True)
        self.bottleneck = UNetBlock(56, latent_dim, use_dropout=True)

        # Decoder: 5 levels
        self.up1 = nn.ConvTranspose2d(latent_dim, 56, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(56 + 56, 56, use_dropout=True)

        self.up2 = nn.ConvTranspose2d(56, 48, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(48 + 48, 48, use_dropout=True)

        self.up3 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(32 + 32, 32, use_dropout=True)

        self.up4 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(24 + 24, 24)

        self.up5 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(16 + 16, 16)

        self.final = nn.Conv2d(16, image_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        z = self.bottleneck(self.pool(e5))
        self._skips = [e1, e2, e3, e4, e5]
        return z

    def decode(self, z):
        e1, e2, e3, e4, e5 = self._skips

        d1 = self.up1(z)
        d1 = self.dec1(torch.cat([d1, e5], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e4], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, e2], dim=1))

        d5 = self.up5(d4)
        d5 = self.dec5(torch.cat([d5, e1], dim=1))

        return self.activation(self.final(d5))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=64, spatial_size=7):
        super().__init__()
        input_dim = latent_dim * spatial_size * spatial_size

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


# Required by main.py
model_class = AdversarialUNetAETest
config_path = "configs/test/adversarial_unet_ae_test.json"
AdversarialUNetAETest.discriminator_class = LatentDiscriminator