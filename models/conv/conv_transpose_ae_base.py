import torch
import torch.nn as nn


def _conv_block(
    in_ch: int,
    out_ch: int,
    *,
    stride: int = 1,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def _deconv_block(
    in_ch: int,
    out_ch: int,
    *,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = [
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
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class ConvTransposeAEBase(nn.Module):
    """
    Fully-convolutional convolutional autoencoder.

    Latent representation:
        [B, latent_channels, image_size / 32, image_size / 32]

    For image_size=224:
        [B, latent_channels, 7, 7]

    This version intentionally removes fc_enc/fc_dec.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_channels = int(config.get("latent_channels", config.get("latent_dim")))

        encoder_channels = list(config["encoder_channels"])
        if len(encoder_channels) != 5:
            raise ValueError("config['encoder_channels'] must contain exactly 5 values.")

        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        c1, c2, c3, c4, c5 = map(int, encoder_channels)
        dropout = float(config.get("dropout", 0.2))

        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_hw = image_size // 32
        self.latent_size = latent_channels * self.latent_hw * self.latent_hw

        self.encoder = nn.Sequential(
            _conv_block(image_channels, c1, stride=2, dropout=0.0),
            _conv_block(c1, c2, stride=2, dropout=0.0),
            _conv_block(c2, c3, stride=2, dropout=0.0),
            _conv_block(c3, c4, stride=2, dropout=dropout),
            _conv_block(c4, c5, stride=2, dropout=dropout),
        )

        self.to_latent = nn.Sequential(
            nn.Conv2d(c5, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, c5, kernel_size=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.decoder = nn.Sequential(
            _deconv_block(c5, c4, dropout=dropout),
            _deconv_block(c4, c3, dropout=dropout),
            _deconv_block(c3, c2, dropout=0.0),
            _deconv_block(c2, c1, dropout=0.0),
            nn.ConvTranspose2d(
                c1,
                image_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.to_latent(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)