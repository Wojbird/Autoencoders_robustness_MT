import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, *, stride: int = 1, dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ConvTransposeAEBase(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        image_channels = int(config["image_channels"])
        ch1 = int(config["ch1"])
        ch2 = int(config["ch2"])
        ch3 = int(config["ch3"])
        ch4 = int(config["ch4"])
        latent_channels = int(config["latent_channels"])
        dropout = float(config.get("dropout", 0.2))

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, ch1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.enc1 = _conv_block(ch1, ch2, stride=2, dropout=0.0)
        self.enc2 = _conv_block(ch2, ch3, stride=2, dropout=0.0)
        self.enc3 = _conv_block(ch3, ch4, stride=2, dropout=0.0)
        self.enc4 = _conv_block(ch4, ch4, stride=2, dropout=dropout)
        self.enc5 = _conv_block(ch4, latent_channels, stride=2, dropout=dropout)

        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.dec1 = _deconv_block(latent_channels, ch4)
        self.dec2 = _deconv_block(ch4, ch3)
        self.dec3 = _deconv_block(ch3, ch2)
        self.dec4 = _deconv_block(ch2, ch1)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                ch1,
                image_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z = self.enc5(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec1(z)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))