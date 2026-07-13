from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


SUPPORTED_ATTACKS = (
    "gaussian_input",
    "gaussian_latent",
    "impulse",
    "rotation",
    "occlusion",
)


@dataclass(frozen=True)
class AttackCondition:
    """One executable robustness-evaluation condition."""

    name: str
    severity: float
    direction: int = 0

    @property
    def is_clean(self) -> bool:
        return self.name == "clean"

    @property
    def is_latent(self) -> bool:
        return self.name == "gaussian_latent"

    @property
    def input_changed(self) -> bool:
        return self.name not in {"clean", "gaussian_latent"}

    @property
    def file_stem(self) -> str:
        severity = format_number_for_filename(self.severity)

        if self.name == "rotation":
            suffix = "plus" if self.direction > 0 else "minus"
            return f"rotation__{severity}__{suffix}"

        return f"{self.name}__{severity}"


def format_number_for_filename(value: float) -> str:
    """Convert a number to a stable, file-name-safe representation."""
    text = f"{float(value):.10g}"
    return text.replace("-", "m").replace(".", "p")


def _normalise_attack_selection(
    selected: Iterable[str] | None,
) -> set[str]:
    if selected is None:
        return set(SUPPORTED_ATTACKS)

    values = {
        str(value).strip().lower()
        for value in selected
        if str(value).strip()
    }

    if not values or "all" in values:
        return set(SUPPORTED_ATTACKS)

    unknown = values.difference(SUPPORTED_ATTACKS)

    if unknown:
        raise ValueError(
            f"Unknown attacks: {sorted(unknown)}. "
            f"Supported attacks: {list(SUPPORTED_ATTACKS)}"
        )

    return values


def build_attack_conditions(
    config: dict,
    selected: Iterable[str] | None = None,
) -> list[AttackCondition]:
    """
    Build all executable conditions.

    The clean condition is always included because it is the
    checkpoint-specific baseline used to calculate degradation under attack.
    """
    chosen = _normalise_attack_selection(selected)
    attacks_config = config.get("attacks", {})

    conditions = [
        AttackCondition(
            name="clean",
            severity=0.0,
            direction=0,
        )
    ]

    for attack_name in SUPPORTED_ATTACKS:
        if attack_name not in chosen:
            continue

        attack_config = attacks_config.get(attack_name, {})
        levels = attack_config.get("levels", [])

        if not levels:
            raise ValueError(
                f"No severity levels configured for attack '{attack_name}'."
            )

        if attack_name == "rotation":
            directions = sorted(
                set(
                    int(value)
                    for value in attack_config.get("directions", [-1, 1])
                )
            )

            if directions != [-1, 1]:
                raise ValueError(
                    "Rotation directions must contain exactly -1 and 1."
                )

            for level in levels:
                level = float(level)

                if level <= 0.0:
                    raise ValueError(
                        "Rotation levels must be positive absolute angles."
                    )

                for direction in directions:
                    conditions.append(
                        AttackCondition(
                            name="rotation",
                            severity=level,
                            direction=direction,
                        )
                    )
        else:
            for level in levels:
                level = float(level)

                if level <= 0.0:
                    raise ValueError(
                        f"Severity for '{attack_name}' must be > 0."
                    )

                conditions.append(
                    AttackCondition(
                        name=attack_name,
                        severity=level,
                        direction=0,
                    )
                )

    return conditions


def stable_seed(
    base_seed: int,
    condition: AttackCondition,
    batch_index: int,
    namespace: str = "",
) -> int:
    """
    Create a stable batch seed independent of Python's randomised hash().

    The evaluation manifest stores batch size, so every checkpoint receives
    exactly the same batches and therefore the same random perturbations.
    """
    payload = (
        f"{int(base_seed)}|"
        f"{condition.name}|"
        f"{condition.severity:.12g}|"
        f"{condition.direction}|"
        f"{int(batch_index)}|"
        f"{namespace}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        % (2**63 - 1)
    )


def _generator(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _gaussian_input(
    x: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(
        x.shape,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )

    return torch.clamp(
        x + sigma * noise,
        min=0.0,
        max=1.0,
    )


def _impulse_noise(
    x: torch.Tensor,
    probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0.0 < probability < 1.0:
        raise ValueError(
            "Impulse probability must be in (0, 1)."
        )

    batch_size, _, height, width = x.shape

    selected_pixels = (
        torch.rand(
            (batch_size, 1, height, width),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        < probability
    )

    salt_mask = (
        torch.rand(
            (batch_size, 1, height, width),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        >= 0.5
    )

    replacement = salt_mask.to(dtype=x.dtype).expand_as(x)

    return torch.where(
        selected_pixels.expand_as(x),
        replacement,
        x,
    )


def _rotation_padding(
    image_size: int,
    angle_degrees: float,
) -> int:
    angle_radians = math.radians(abs(angle_degrees))
    half_size = image_size / 2.0

    required_half_size = half_size * (
        abs(math.cos(angle_radians))
        + abs(math.sin(angle_radians))
    )

    return max(
        1,
        int(math.ceil(required_half_size - half_size)) + 2,
    )


def _rotate_with_reflection_padding(
    x: torch.Tensor,
    angle_degrees: float,
) -> torch.Tensor:
    """
    Rotate after reflection padding and crop back to the original resolution.

    This avoids large black triangular regions at image corners.
    """
    _, _, height, width = x.shape

    if height != width:
        raise ValueError(
            "Rotation implementation expects square images."
        )

    padding = _rotation_padding(height, angle_degrees)

    padded = F.pad(
        x,
        pad=(padding, padding, padding, padding),
        mode="reflect",
    )

    rotated = TF.rotate(
        padded,
        angle=float(angle_degrees),
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0.0,
    )

    attacked = TF.center_crop(
        rotated,
        output_size=[height, width],
    )

    return torch.clamp(
        attacked,
        min=0.0,
        max=1.0,
    )


def _occlusion(
    x: torch.Tensor,
    area_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0.0 < area_fraction < 1.0:
        raise ValueError(
            "Occlusion area fraction must be in (0, 1)."
        )

    batch_size, _, height, width = x.shape

    side = max(
        1,
        int(
            round(
                math.sqrt(
                    area_fraction * height * width
                )
            )
        ),
    )

    side = min(side, height, width)

    top_positions = torch.randint(
        low=0,
        high=height - side + 1,
        size=(batch_size,),
        generator=generator,
        device=x.device,
    )

    left_positions = torch.randint(
        low=0,
        high=width - side + 1,
        size=(batch_size,),
        generator=generator,
        device=x.device,
    )

    attacked = x.clone()
    channel_means = x.mean(dim=(2, 3), keepdim=True)

    for sample_index in range(batch_size):
        top = int(top_positions[sample_index].item())
        left = int(left_positions[sample_index].item())

        attacked[
            sample_index,
            :,
            top : top + side,
            left : left + side,
        ] = channel_means[sample_index]

    return attacked


def apply_input_attack(
    x: torch.Tensor,
    condition: AttackCondition,
    *,
    base_seed: int,
    batch_index: int,
) -> torch.Tensor:
    """
    Apply an input-space attack.

    For clean and gaussian_latent conditions the image stays unchanged.
    """
    if condition.name in {"clean", "gaussian_latent"}:
        return x

    generator = _generator(
        x.device,
        stable_seed(
            base_seed,
            condition,
            batch_index,
            namespace="input",
        ),
    )

    if condition.name == "gaussian_input":
        return _gaussian_input(
            x,
            condition.severity,
            generator,
        )

    if condition.name == "impulse":
        return _impulse_noise(
            x,
            condition.severity,
            generator,
        )

    if condition.name == "rotation":
        return _rotate_with_reflection_padding(
            x,
            angle_degrees=(
                condition.direction * condition.severity
            ),
        )

    if condition.name == "occlusion":
        return _occlusion(
            x,
            condition.severity,
            generator,
        )

    raise ValueError(
        f"Unsupported input attack: {condition.name}"
    )


def apply_latent_attack(
    z: torch.Tensor,
    condition: AttackCondition,
    *,
    base_seed: int,
    batch_index: int,
) -> torch.Tensor:
    """
    Apply relative Gaussian latent noise, matching noisy_latent training:

        z_attacked = z + alpha * std(z) * epsilon
    """
    if condition.name != "gaussian_latent":
        return z

    generator = _generator(
        z.device,
        stable_seed(
            base_seed,
            condition,
            batch_index,
            namespace="latent",
        ),
    )

    noise = torch.randn(
        z.shape,
        device=z.device,
        dtype=z.dtype,
        generator=generator,
    )

    scale = (
        z.detach()
        .flatten(start_dim=1)
        .std(dim=1, unbiased=False)
        .view(-1, 1, 1, 1)
        .clamp_min(1e-6)
    )

    return z + condition.severity * scale * noise