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

class AdversarialUNetAE512(nn.Module):
    discriminator_class = None

    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 512, "This model is designed for latent_dim=512"

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = UNetBlock(image_channels, 64)   # 64×224×224
        self.enc2 = UNetBlock(64, 128)  # 128×112×112
        self.enc3 = UNetBlock(128, 256, use_dropout=True)   # 256×56×56
        self.enc4 = UNetBlock(256, 384, use_dropout=True)   # 384×28×28
        self.enc5 = UNetBlock(384, 448, use_dropout=True)   # 448×14×14

        # Bottleneck
        self.bottleneck = UNetBlock(448, latent_dim, use_dropout=True)  # 512×7×7

        # Decoder
        self.up1 = nn.ConvTranspose2d(latent_dim, 448, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(448 + 448, 448, use_dropout=True) # 448×14×14

        self.up2 = nn.ConvTranspose2d(448, 384, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(384 + 384, 384, use_dropout=True) # 384×28×28

        self.up3 = nn.ConvTranspose2d(384, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256 + 256, 256, use_dropout=True) # 256×56×56

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(128 + 128, 128)   # 128×112×112

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(64 + 64, 64)  # 64×224×224

        self.final = nn.Conv2d(64, image_channels, kernel_size=1)   # 3×224×224
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

        out = self.activation(self.final(d5))
        return out

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

class ImageDiscriminator(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=1),  # 64×112×112
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128×56×56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256×28×28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 512×14×14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1), # 512 × 1 × 1
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model_class = AdversarialUNetAE512
config_path = "configs/adversarial_unet_ae_512.json"
AdversarialUNetAE512.discriminator_class = ImageDiscriminator