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

class UNetAE256(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 3136, "Model designed for latent_dim=3136"

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
        z_map = self.bottleneck(self.pool(e5))

        z_vec = self.fc_enc(z_map.flatten(1))  # (B, latent_dim)
        return z_vec

    def decode(self, z_vec):
        b = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(b, 256, 7, 7)

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

model_class = UNetAE256
config_path = "configs/unet/unet_ae_256.json"