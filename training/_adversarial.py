import torch
import torch.nn as nn


def unwrap_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def is_adversarial_model(model: nn.Module) -> bool:
    return hasattr(model, "discriminator_class") and getattr(model, "discriminator_class") is not None


def build_discriminator(model: nn.Module, cfg: dict, device: torch.device):
    if not is_adversarial_model(model):
        return None, None, None, 0.0

    discriminator = model.discriminator_class(cfg).to(device)
    disc_lr = float(cfg.get("disc_learning_rate", cfg["learning_rate"]))
    disc_wd = float(cfg.get("disc_weight_decay", cfg.get("weight_decay", 0.0)))
    adv_weight = float(cfg.get("adversarial_weight", getattr(model, "adv_weight", 1e-3)))

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=disc_lr,
        weight_decay=disc_wd,
        betas=(0.5, 0.999),
    )
    bce = nn.BCEWithLogitsLoss()
    return discriminator, optimizer_d, bce, adv_weight


def discriminator_step(
    discriminator: nn.Module,
    optimizer_d,
    bce,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
):
    optimizer_d.zero_grad(set_to_none=True)

    real_logits = discriminator(x_real)
    fake_logits = discriminator(x_fake.detach())

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)

    loss_real = bce(real_logits, real_targets)
    loss_fake = bce(fake_logits, fake_targets)
    loss_d = 0.5 * (loss_real + loss_fake)

    loss_d.backward()
    optimizer_d.step()
    return float(loss_d.item())


def generator_adv_loss(
    discriminator: nn.Module,
    bce,
    x_fake: torch.Tensor,
) -> torch.Tensor:
    fake_logits = discriminator(x_fake)
    real_targets = torch.ones_like(fake_logits)
    return bce(fake_logits, real_targets)