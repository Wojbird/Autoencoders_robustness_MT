import torch
import torch.nn as nn

from models.unet.unet_ae_base import UNetAEBase


def _num_groups(channels: int) -> int:
    for g in (8, 4, 2):
        if channels % g == 0:
            return g
    return 1


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=_num_groups(channels), num_channels=channels)


class ImageDiscriminator(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config.get("image_channels", 3))
        disc_channels = list(config.get("disc_channels", [32, 64, 96, 128]))

        if len(disc_channels) != 4:
            raise ValueError("config['disc_channels'] must contain exactly 4 values.")

        c1, c2, c3, c4 = map(int, disc_channels)

        self.net = nn.Sequential(
            nn.Conv2d(
                image_channels,
                c1,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            _norm(c2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                c2,
                c3,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            _norm(c3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                c3,
                c4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            _norm(c4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdversarialAEBase(UNetAEBase):
    """
    Adversarial autoencoder generator based on UNetAEBase without skip connections.
    """

    discriminator_class = ImageDiscriminator

    def __init__(self, config: dict):
        super().__init__(config)
        self.adv_weight = float(config.get("adversarial_weight", 1e-3))