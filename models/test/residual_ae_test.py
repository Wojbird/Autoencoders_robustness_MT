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
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


def _make_encoder_stage(
    in_channels: int,
    out_channels: int,
    *,
    use_dropout: bool,
    dropout: float,
) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if use_dropout and dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    layers.append(ResBlock(out_channels))
    return nn.Sequential(*layers)


def _make_decoder_stage(
    in_channels: int,
    out_channels: int,
    *,
    use_dropout: bool,
    dropout: float,
) -> nn.Sequential:
    layers = [
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if use_dropout and dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    layers.append(ResBlock(out_channels))
    return nn.Sequential(*layers)


class ResidualAETest(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))
        latent_dim = int(config["latent_dim"])
        dropout = float(config.get("dropout", 0.1))

        pre_channels = list(config.get("pre_channels", [8, 16]))
        enc_channels = list(config.get("enc_channels", [24, 32, 48, 56, 64]))

        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")
        if len(pre_channels) != 2:
            raise ValueError("pre_channels must contain exactly 2 values.")
        if len(enc_channels) != 5:
            raise ValueError("enc_channels must contain exactly 5 values.")

        self._latent_hw = image_size // 32
        self._latent_channels = enc_channels[-1]

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, pre_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(pre_channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(pre_channels[0], pre_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(pre_channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.enc1 = _make_encoder_stage(pre_channels[1], enc_channels[0], use_dropout=False, dropout=dropout)
        self.enc2 = _make_encoder_stage(enc_channels[0], enc_channels[1], use_dropout=True, dropout=dropout)
        self.enc3 = _make_encoder_stage(enc_channels[1], enc_channels[2], use_dropout=True, dropout=dropout)
        self.enc4 = _make_encoder_stage(enc_channels[2], enc_channels[3], use_dropout=True, dropout=dropout)
        self.enc5 = _make_encoder_stage(enc_channels[3], enc_channels[4], use_dropout=True, dropout=dropout)

        flat_dim = self._latent_channels * self._latent_hw * self._latent_hw
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        dec_channels = list(reversed(enc_channels[:-1])) + [pre_channels[1]]

        self.dec1 = _make_decoder_stage(enc_channels[4], dec_channels[0], use_dropout=True, dropout=dropout)
        self.dec2 = _make_decoder_stage(dec_channels[0], dec_channels[1], use_dropout=True, dropout=dropout)
        self.dec3 = _make_decoder_stage(dec_channels[1], dec_channels[2], use_dropout=True, dropout=dropout)
        self.dec4 = _make_decoder_stage(dec_channels[2], dec_channels[3], use_dropout=False, dropout=dropout)
        self.dec5 = _make_decoder_stage(dec_channels[3], dec_channels[4], use_dropout=False, dropout=dropout)

        self.final = nn.Conv2d(pre_channels[1], image_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z_map = self.enc5(x)
        return self.fc_enc(z_map.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        x = self.fc_dec(z).view(b, self._latent_channels, self._latent_hw, self._latent_hw)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        return self.activation(self.final(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


model_class = ResidualAETest
config_path = "configs/test/residual_ae_test.json"