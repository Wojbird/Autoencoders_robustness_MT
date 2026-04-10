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
        image_size = int(config.get("image_size", 224))
        latent_dim = int(config["latent_dim"])

        encoder_channels = list(config["encoder_channels"])
        if len(encoder_channels) != 4:
            raise ValueError("config['encoder_channels'] must contain exactly 4 values.")

        ch1, ch2, ch3, ch4 = map(int, encoder_channels)
        bottleneck_channels = int(config["bottleneck_channels"])
        dropout = float(config.get("dropout", 0.2))

        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        self._latent_hw = image_size // 32
        self._bottleneck_channels = bottleneck_channels

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, ch1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.enc1 = _conv_block(ch1, ch2, stride=2, dropout=0.0)
        self.enc2 = _conv_block(ch2, ch3, stride=2, dropout=0.0)
        self.enc3 = _conv_block(ch3, ch4, stride=2, dropout=dropout)
        self.enc4 = _conv_block(ch4, ch4, stride=2, dropout=dropout)
        self.enc5 = _conv_block(ch4, bottleneck_channels, stride=2, dropout=dropout)

        flat_dim = bottleneck_channels * self._latent_hw * self._latent_hw
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        self.dec1 = _deconv_block(bottleneck_channels, ch4)
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

    def _encode_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_map = self._encode_map(x)
        return self.fc_enc(z_map.flatten(1))

    def decode(self, z_vec: torch.Tensor) -> torch.Tensor:
        batch_size = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(
            batch_size,
            self._bottleneck_channels,
            self._latent_hw,
            self._latent_hw,
        )

        x = self.dec1(z_map)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)