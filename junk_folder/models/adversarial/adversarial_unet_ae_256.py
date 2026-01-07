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
    discriminator_class = None

    def __init__(self, config: dict):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 256, "Model designed for latent_dim=256"

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = UNetBlock(image_channels, 32)   # 32×224×224
        self.enc2 = UNetBlock(32, 64)   # 64×112×112
        self.enc3 = UNetBlock(64, 96, use_dropout=True) # 96×56×56
        self.enc4 = UNetBlock(96, 128, use_dropout=True)    # 128×28×28
        self.enc5 = UNetBlock(128, 192, use_dropout=True)   # 192×14×14

        # Bottleneck
        self.bottleneck = UNetBlock(192, latent_dim, use_dropout=True)  # 256×7×7

        # Decoder
        self.up1 = nn.ConvTranspose2d(latent_dim, 192, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(192, 192, use_dropout=True) # 192×14×14

        self.up2 = nn.ConvTranspose2d(192, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 128, use_dropout=True) # 128×28×28

        self.up3 = nn.ConvTranspose2d(128, 96, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(96, 96, use_dropout=True)    # 96×56×56

        self.up4 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(64, 64)  # 64×112×112

        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(32, 32)  # 32×224×224

        self.final = nn.Conv2d(32, image_channels, kernel_size=1)   # 3×224×224
        self.activation = nn.Sigmoid()

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        z = self.bottleneck(self.pool(e5))
        return z

    def decode(self, z):

        d1 = self.up1(z)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        d4 = self.up4(d3)
        d5 = self.up5(d4)
        return self.activation(self.final(d5))

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

model_class = AdversarialUNetAE256
config_path = "configs/adversarial/adversarial_unet_ae_256.json"
AdversarialUNetAE256.discriminator_class = ImageDiscriminator