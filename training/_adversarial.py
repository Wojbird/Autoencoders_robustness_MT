from contextlib import contextmanager
from collections.abc import Iterator

import torch
import torch.nn as nn

from utils.latent import unwrap_tensor


def is_adversarial_model(model: nn.Module) -> bool:
    discriminator_class = getattr(model, "discriminator_class", None)
    return callable(discriminator_class)


def build_discriminator(
    model: nn.Module,
    cfg: dict,
    device: torch.device,
) -> tuple[
    nn.Module | None,
    torch.optim.Optimizer | None,
    nn.Module | None,
    float,
]:
    if not is_adversarial_model(model):
        return None, None, None, 0.0

    discriminator_class = model.discriminator_class
    discriminator = discriminator_class(cfg).to(device)

    discriminator_learning_rate = float(
        cfg.get("disc_learning_rate", cfg["learning_rate"])
    )
    discriminator_weight_decay = float(
        cfg.get("disc_weight_decay", cfg.get("weight_decay", 0.0))
    )
    adversarial_weight = float(
        cfg.get(
            "adversarial_weight",
            getattr(model, "adv_weight", 1e-3),
        )
    )

    if discriminator_learning_rate <= 0.0:
        raise ValueError("disc_learning_rate must be positive.")
    if discriminator_weight_decay < 0.0:
        raise ValueError("disc_weight_decay must be non-negative.")
    if adversarial_weight < 0.0:
        raise ValueError("adversarial_weight must be non-negative.")

    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_learning_rate,
        weight_decay=discriminator_weight_decay,
        betas=(0.5, 0.999),
    )

    return (
        discriminator,
        optimizer,
        nn.BCEWithLogitsLoss(),
        adversarial_weight,
    )


def discriminator_step(
    discriminator: nn.Module,
    optimizer_d: torch.optim.Optimizer,
    bce: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
) -> float:
    optimizer_d.zero_grad(set_to_none=True)

    real_logits = discriminator(x_real)
    fake_logits = discriminator(x_fake.detach())

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)

    real_loss = bce(real_logits, real_targets)
    fake_loss = bce(fake_logits, fake_targets)
    discriminator_loss = 0.5 * (real_loss + fake_loss)

    discriminator_loss.backward()
    optimizer_d.step()

    return float(discriminator_loss.item())


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    original_requires_grad = [
        parameter.requires_grad
        for parameter in module.parameters()
    ]

    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

        yield

    finally:
        for parameter, requires_grad in zip(
            module.parameters(),
            original_requires_grad,
        ):
            parameter.requires_grad_(requires_grad)


def generator_adv_loss(
    discriminator: nn.Module,
    bce: nn.Module,
    x_fake: torch.Tensor,
) -> torch.Tensor:
    # Freeze discriminator weights while preserving the gradient path from
    # discriminator output to x_fake and therefore to the generator.
    with frozen_parameters(discriminator):
        fake_logits = discriminator(x_fake)
        real_targets = torch.ones_like(fake_logits)
        return bce(fake_logits, real_targets)