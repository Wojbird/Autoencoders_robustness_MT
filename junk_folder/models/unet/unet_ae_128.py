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

class UNetAE128(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 128, "Model designed for latent_dim=128"

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = UNetBlock(image_channels, 32)   # 32×224×224
        self.enc2 = UNetBlock(32, 48)   # 48×112×112
        self.enc3 = UNetBlock(48, 64, use_dropout=True) # 64×56×56
        self.enc4 = UNetBlock(64, 96, use_dropout=True)    # 96×28×28
        self.enc5 = UNetBlock(96, 112, use_dropout=True)   # 112×14×14

        # Bottleneck
        self.bottleneck = UNetBlock(112, latent_dim, use_dropout=True)  # 128×7×7

        # Decoder
        self.up1 = nn.ConvTranspose2d(latent_dim, 112, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(112, 112, use_dropout=True) # 112×14×14

        self.up2 = nn.ConvTranspose2d(112, 96, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(96, 96, use_dropout=True) # 96×28×28

        self.up3 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(64, 64, use_dropout=True)    # 64×56×56

        self.up4 = nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(48, 48)  # 48×112×112

        self.up5 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(32, 32)  # 32×224×224

        self.final = nn.Conv2d(32, image_channels, kernel_size=1)   # 3×224×224
        self.activation = nn.Tanh()

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
        out = self.activation(self.final(d5))
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

model_class = UNetAE128
config_path = "configs/unet/unet_ae_128.json"