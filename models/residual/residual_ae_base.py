import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


def make_encoder_stage(
    in_channels: int,
    out_channels: int,
    use_dropout: bool,
    dropout_p: float
) -> nn.Sequential:
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    ]

    if use_dropout and dropout_p > 0.0:
        layers.append(nn.Dropout2d(dropout_p))

    layers.append(ResBlock(out_channels))
    return nn.Sequential(*layers)


def make_decoder_stage(
    in_channels: int,
    out_channels: int,
    use_dropout: bool,
    dropout_p: float
) -> nn.Sequential:
    layers = [
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    ]

    if use_dropout and dropout_p > 0.0:
        layers.append(nn.Dropout2d(dropout_p))

    layers.append(ResBlock(out_channels))
    return nn.Sequential(*layers)


class ResidualAEBase(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_dim = int(config["latent_dim"])

        pre_channels = list(config["pre_channels"])
        enc_channels = list(config["enc_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(pre_channels) != 2:
            raise ValueError("config['pre_channels'] must contain exactly 2 values.")
        if len(enc_channels) != 5:
            raise ValueError("config['enc_channels'] must contain exactly 5 values.")
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        self._latent_hw = image_size // 32
        self._latent_channels = enc_channels[-1]

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, pre_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(pre_channels[0]),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(pre_channels[0], pre_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(pre_channels[1]),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Encoder: enc1 bez dropout, enc2-enc5 z dropout
        self.enc1 = make_encoder_stage(pre_channels[1], enc_channels[0], False, dropout)
        self.enc2 = make_encoder_stage(enc_channels[0], enc_channels[1], True, dropout)
        self.enc3 = make_encoder_stage(enc_channels[1], enc_channels[2], True, dropout)
        self.enc4 = make_encoder_stage(enc_channels[2], enc_channels[3], True, dropout)
        self.enc5 = make_encoder_stage(enc_channels[3], enc_channels[4], True, dropout)

        flat_dim = self._latent_channels * self._latent_hw * self._latent_hw
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        dec_channels = list(reversed(enc_channels[:-1])) + [pre_channels[1]]

        # Decoder: dec1-dec3 z dropout, dec4-dec5 bez dropout
        self.dec1 = make_decoder_stage(enc_channels[4], dec_channels[0], True, dropout)
        self.dec2 = make_decoder_stage(dec_channels[0], dec_channels[1], True, dropout)
        self.dec3 = make_decoder_stage(dec_channels[1], dec_channels[2], True, dropout)
        self.dec4 = make_decoder_stage(dec_channels[2], dec_channels[3], False, dropout)
        self.dec5 = make_decoder_stage(dec_channels[3], dec_channels[4], False, dropout)

        self.final = nn.Conv2d(pre_channels[1], image_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z_map = self.enc5(x)

        z_vec = self.fc_enc(z_map.flatten(1))
        return z_vec

    def decode(self, z_vec: torch.Tensor) -> torch.Tensor:
        batch_size = z_vec.size(0)
        z_map = self.fc_dec(z_vec).view(
            batch_size,
            self._latent_channels,
            self._latent_hw,
            self._latent_hw
        )

        x = self.dec1(z_map)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        return self.activation(self.final(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)