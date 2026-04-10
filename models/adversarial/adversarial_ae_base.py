import torch
import torch.nn as nn

from models.unet.unet_ae_base import UNetAEBase


class ImageDiscriminator(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config.get("image_channels", 3))
        disc_channels = list(config.get("disc_channels", [32, 64, 96, 128]))

        if len(disc_channels) != 4:
            raise ValueError("config['disc_channels'] must contain exactly 4 values.")

        c1, c2, c3, c4 = map(int, disc_channels)

        self.net = nn.Sequential(
            nn.Conv2d(image_channels, c1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdversarialAEBase(UNetAEBase):
    discriminator_class = ImageDiscriminator

    def __init__(self, config: dict):
        super().__init__(config)
        self.adv_weight = float(config.get("adversarial_weight", 1e-3))