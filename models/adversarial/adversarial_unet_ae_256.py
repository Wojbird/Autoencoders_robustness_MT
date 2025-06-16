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
            layers.insert(4, nn.Dropout2d(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class AdversarialUNetAE256(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 256, "Model designed for latent_dim=256"

        self.pool = nn.MaxPool2d(2)

        # Encoder blocks (5 levels)
        self.enc1 = UNetBlock(image_channels, 32)          # H x W
        self.enc2 = UNetBlock(32, 64)                       # H/2 x W/2
        self.enc3 = UNetBlock(64, 96, use_dropout=True)    # H/4 x W/4
        self.enc4 = UNetBlock(96, 128, use_dropout=True)   # H/8 x W/8
        self.enc5 = UNetBlock(128, 192, use_dropout=True)  # H/16 x W/16

        # Bottleneck (latent representation)
        self.bottleneck = UNetBlock(192, latent_dim, use_dropout=True)  # H/32 x W/32

        # Decoder blocks (5 levels, with skip connections)
        self.up1 = nn.ConvTranspose2d(latent_dim, 192, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(192 + 192, 192, use_dropout=True)

        self.up2 = nn.ConvTranspose2d(192, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128 + 128, 128, use_dropout=True)

        self.up3 = nn.ConvTranspose2d(128, 96, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(96 + 96, 96, use_dropout=True)

        self.up4 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(64 + 64, 64)

        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(32 + 32, 32)

        # Final conv + activation
        self.final = nn.Conv2d(32, image_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def encode(self, x):
        e1 = self.enc1(x)              # H x W
        e2 = self.enc2(self.pool(e1)) # H/2 x W/2
        e3 = self.enc3(self.pool(e2)) # H/4 x W/4
        e4 = self.enc4(self.pool(e3)) # H/8 x W/8
        e5 = self.enc5(self.pool(e4)) # H/16 x W/16
        z = self.bottleneck(self.pool(e5))  # H/32 x W/32
        self._skips = [e1, e2, e3, e4, e5]
        return z

    def decode(self, z):
        e1, e2, e3, e4, e5 = self._skips

        d1 = self.up1(z)                      # H/16 x W/16
        d1 = self.dec1(torch.cat([d1, e5], dim=1))

        d2 = self.up2(d1)                     # H/8 x W/8
        d2 = self.dec2(torch.cat([d2, e4], dim=1))

        d3 = self.up3(d2)                     # H/4 x W/4
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d4 = self.up4(d3)                     # H/2 x W/2
        d4 = self.dec4(torch.cat([d4, e2], dim=1))

        d5 = self.up5(d4)                     # H x W
        d5 = self.dec5(torch.cat([d5, e1], dim=1))

        out = self.activation(self.final(d5))
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=256, spatial_size=7):
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
model_class = AdversarialUNetAE256
config_path = "configs/adversarial_unet_ae_256.json"
AdversarialUNetAE256.discriminator_class = LatentDiscriminator