import math
from collections.abc import Callable

import torch
import torch.nn as nn


LatentTransform = Callable[[torch.Tensor], torch.Tensor]


def unwrap_tensor(value):
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("Cannot unwrap an empty model output.")
        return value[0]

    return value


def add_gaussian_latent_noise(
    z: torch.Tensor,
    noise_std: float,
    *,
    relative: bool = True,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    noise_std = float(noise_std)

    if not math.isfinite(noise_std):
        raise ValueError("noise_std must be finite.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if noise_std == 0.0:
        return z

    if not torch.is_tensor(z):
        raise TypeError("The latent representation must be a torch.Tensor.")
    if not z.is_floating_point():
        raise TypeError("The latent representation must use a floating dtype.")
    if z.ndim < 2:
        raise ValueError(
            "Expected a batched latent tensor with at least two dimensions, "
            f"received shape {tuple(z.shape)}."
        )

    if noise is None:
        noise = torch.randn_like(z)
    else:
        if not torch.is_tensor(noise):
            raise TypeError("noise must be a torch.Tensor.")
        if noise.shape != z.shape:
            raise ValueError(
                f"Noise shape {tuple(noise.shape)} does not match "
                f"latent shape {tuple(z.shape)}."
            )
        if noise.device != z.device:
            raise ValueError("noise and latent tensors must use the same device.")
        if noise.dtype != z.dtype:
            raise ValueError("noise and latent tensors must use the same dtype.")

    if not relative:
        return z + noise_std * noise

    broadcast_shape = (z.shape[0],) + (1,) * (z.ndim - 1)

    scale = (
        z.detach()
        .flatten(start_dim=1)
        .std(dim=1, unbiased=False)
        .view(broadcast_shape)
        .clamp_min(1e-6)
    )

    return z + noise_std * scale * noise


def _apply_transform(
    z: torch.Tensor,
    transform: LatentTransform,
) -> torch.Tensor:
    transformed = transform(z)

    if not torch.is_tensor(transformed):
        raise TypeError("A latent transform must return a torch.Tensor.")
    if transformed.shape != z.shape:
        raise ValueError(
            f"A latent transform changed the shape from {tuple(z.shape)} "
            f"to {tuple(transformed.shape)}."
        )

    return transformed


def forward_with_latent_transform(
    model: nn.Module,
    x: torch.Tensor,
    transform: LatentTransform,
) -> torch.Tensor:
    if not callable(transform):
        raise TypeError("transform must be callable.")

    custom_forward = getattr(
        model,
        "forward_with_latent_transform",
        None,
    )

    if callable(custom_forward):
        def checked_transform(z: torch.Tensor) -> torch.Tensor:
            return _apply_transform(z, transform)

        return unwrap_tensor(custom_forward(x, checked_transform))

    encode = getattr(model, "encode", None)
    decode = getattr(model, "decode", None)

    if callable(encode) and callable(decode):
        z = unwrap_tensor(encode(x))
        z = _apply_transform(z, transform)
        return unwrap_tensor(decode(z))

    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)

    if callable(encoder) and callable(decoder):
        z = unwrap_tensor(encoder(x))
        z = _apply_transform(z, transform)
        return unwrap_tensor(decoder(z))

    raise AttributeError(
        f"{model.__class__.__name__} must implement "
        "forward_with_latent_transform(), encode()/decode(), "
        "or encoder/decoder."
    )