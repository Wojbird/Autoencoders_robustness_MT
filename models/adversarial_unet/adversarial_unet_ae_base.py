from __future__ import annotations

import torch
import torch.nn as nn

from models.common.blocks import group_norm
from models.unet.unet_ae_base import UNetAEBase


class ImageDiscriminator(nn.Module):
    """Small image discriminator; its parameters are reported separately."""

    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config.get("image_channels", 3))
        channels = [int(value) for value in config.get("disc_channels", [32, 64, 96, 128])]
        if len(channels) != 4:
            raise ValueError("config['disc_channels'] must contain exactly 4 values.")
        if any(value <= 0 for value in channels):
            raise ValueError("All disc_channels values must be positive.")
        if any(left >= right for left, right in zip(channels, channels[1:])):
            raise ValueError("disc_channels must be strictly increasing.")

        c1, c2, c3, c4 = channels
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            group_norm(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            group_norm(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            group_norm(c4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdversarialUNetAEBase(UNetAEBase):
    """True U-Net generator trained with reconstruction and adversarial_unet losses."""

    discriminator_class = ImageDiscriminator

    def __init__(self, config: dict):
        super().__init__(config)
        self.adv_weight = float(config.get("adversarial_weight", 1e-3))
        if self.adv_weight < 0.0:
            raise ValueError("adversarial_weight must be non-negative.")
