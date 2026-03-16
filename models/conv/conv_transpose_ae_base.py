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