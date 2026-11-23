from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Iterable

import torch


MAIN_EXPERIMENT_SIZES = frozenset({
    16,
    32,
    64,
    128,
    256,
})


FAMILY_DIRECTORIES = {
    "conv": "conv",
    "residual": "residual",
    "u_shaped": "u_shaped",
    "unet": "unet",
    "adversarial_unet": "adversarial_unet",
    "vqv": "vqv",
}


FAMILY_ALIASES = {
    "conv": "conv",
    "convolutional": "conv",
    "conv_transpose": "conv",
    "residual": "residual",
    "res": "residual",
    "u_shaped": "u_shaped",
    "ushaped": "u_shaped",
    "u-shaped": "u_shaped",
    "unet": "unet",
    "unet": "unet",
    "adversarial_unet": "adversarial_unet",
    "adversarial_unet": "adversarial_unet",
    "adv": "adversarial_unet",
    "gan": "adversarial_unet",
    "vqv": "vqv",
    "vq": "vqv",
    "vqvae": "vqv",
    "vq_v_ae": "vqv",
}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    family: str
    latent_channels: int
    latent_elements: int
    module_name: str
    module_path: Path
    config_path: Path
    model_class: type[torch.nn.Module]
    config: dict
    # Kept only for compatibility with the current robustness CLI.
    # Full-retraining experiments intentionally do not use overrides.
    config_override: dict

    def checkpoint_path(
        self,
        checkpoint_root: Path,
        checkpoint_dataset: str,
        training_variant: str,
    ) -> Path:
        return (
            checkpoint_root
            / self.model_name
            / checkpoint_dataset
            / training_variant
            / f"{self.model_name}_{training_variant}_best.pt"
        )

    @property
    def supported_training_variants(self) -> tuple[str, ...]:
        if self.family == "vqv":
            return (
                "clean",
                "noisy",
            )

        return (
            "clean",
            "noisy",
            "noisy_latent",
        )

    @property
    def uses_checkpoint_override(self) -> bool:
        return False


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Configuration root must be a JSON object: {path}"
        )

    return payload


def _normalise_selectors(
    selectors: Iterable[str] | None,
) -> list[str]:
    if selectors is None:
        return ["all"]

    result = [
        str(value).strip().lower()
        for value in selectors
        if str(value).strip()
    ]

    return result or ["all"]


def _module_name_from_path(
    project_root: Path,
    module_path: Path,
) -> str:
    relative_path = (
        module_path
        .with_suffix("")
        .relative_to(project_root)
    )

    return ".".join(relative_path.parts)


def _validate_model_config(
    *,
    config: dict,
    config_path: Path,
    expected_family: str,
    filename_size: int,
) -> tuple[str, int, int]:
    required_keys = {
        "name",
        "family",
        "image_size",
        "image_channels",
        "latent_channels",
    }
    missing_keys = required_keys.difference(config)

    if missing_keys:
        raise KeyError(
            f"Configuration {config_path} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    model_name = str(config["name"])
    config_family = str(config["family"]).strip().lower()

    if config_family != expected_family:
        raise ValueError(
            f"Configuration family mismatch in {config_path}: "
            f"found {config_family!r}, expected {expected_family!r}."
        )

    latent_channels = int(config["latent_channels"])

    if latent_channels != filename_size:
        raise ValueError(
            f"Latent size mismatch in {config_path}: filename indicates "
            f"{filename_size}, config contains {latent_channels}."
        )

    if latent_channels not in MAIN_EXPERIMENT_SIZES:
        raise ValueError(
            f"Unsupported main-experiment latent size: {latent_channels}."
        )

    image_size = int(config["image_size"])
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError(
            f"image_size must be positive and divisible by 32: {config_path}"
        )

    image_channels = int(config["image_channels"])
    if image_channels <= 0:
        raise ValueError(
            f"image_channels must be positive: {config_path}"
        )

    latent_spatial_size = int(
        config.get("latent_spatial_size", image_size // 32)
    )
    expected_spatial_size = image_size // 32

    if latent_spatial_size != expected_spatial_size:
        raise ValueError(
            f"latent_spatial_size mismatch in {config_path}: found "
            f"{latent_spatial_size}, expected {expected_spatial_size}."
        )

    latent_elements = latent_channels * latent_spatial_size**2

    configured_latent_size = int(
        config.get("latent_size", latent_elements)
    )
    if configured_latent_size != latent_elements:
        raise ValueError(
            f"latent_size mismatch in {config_path}: found "
            f"{configured_latent_size}, expected {latent_elements}."
        )

    return model_name, latent_channels, latent_elements


def discover_model_specs(
    project_root: Path,
    selectors: Iterable[str] | None = None,
) -> list[ModelSpec]:
    project_root = project_root.resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    models_root = project_root / "models"
    specs: list[ModelSpec] = []
    seen_model_names: set[str] = set()

    for family, directory_name in FAMILY_DIRECTORIES.items():
        family_root = models_root / directory_name

        if not family_root.is_dir():
            continue

        for module_path in sorted(family_root.glob("*.py")):
            if (
                module_path.name.endswith("_base.py")
                or module_path.name == "__init__.py"
            ):
                continue

            suffix = module_path.stem.rsplit("_", maxsplit=1)[-1]

            if not suffix.isdigit():
                continue

            filename_size = int(suffix)
            if filename_size not in MAIN_EXPERIMENT_SIZES:
                continue

            module_name = _module_name_from_path(
                project_root,
                module_path,
            )
            module = importlib.import_module(module_name)

            model_class = getattr(module, "model_class", None)
            config_value = getattr(module, "config_path", None)

            if model_class is None or config_value is None:
                raise AttributeError(
                    f"{module_name} must define model_class and config_path."
                )

            if (
                not isinstance(model_class, type)
                or not issubclass(model_class, torch.nn.Module)
            ):
                raise TypeError(
                    f"model_class in {module_name} must be an nn.Module class."
                )

            config_path = (project_root / config_value).resolve()
            config = _load_json(config_path)

            (
                model_name,
                latent_channels,
                latent_elements,
            ) = _validate_model_config(
                config=config,
                config_path=config_path,
                expected_family=family,
                filename_size=filename_size,
            )

            if model_name in seen_model_names:
                raise ValueError(
                    f"Duplicate model name discovered: {model_name}"
                )

            seen_model_names.add(model_name)

            specs.append(
                ModelSpec(
                    model_name=model_name,
                    family=family,
                    latent_channels=latent_channels,
                    latent_elements=latent_elements,
                    module_name=module_name,
                    module_path=module_path,
                    config_path=config_path,
                    model_class=model_class,
                    config=deepcopy(config),
                    config_override={},
                )
            )

    normalised_selectors = _normalise_selectors(selectors)

    if "all" in normalised_selectors:
        return sorted(
            specs,
            key=lambda item: (
                item.family,
                item.latent_channels,
            ),
        )

    selected: list[ModelSpec] = []

    for spec in specs:
        for selector in normalised_selectors:
            canonical_family = FAMILY_ALIASES.get(selector)
            exact_model_match = selector == spec.model_name.lower()
            family_match = canonical_family == spec.family

            if exact_model_match or family_match:
                selected.append(spec)
                break

    if not selected:
        available_models = sorted(spec.model_name for spec in specs)
        available_families = sorted(FAMILY_DIRECTORIES)

        raise ValueError(
            "No model matched selectors "
            f"{normalised_selectors}. Available families: "
            f"{available_families}. Available models: {available_models}."
        )

    return sorted(
        selected,
        key=lambda item: (
            item.family,
            item.latent_channels,
        ),
    )


def _extract_state_dict(payload):
    if not isinstance(payload, dict):
        return payload

    for key in (
        "model_state_dict",
        "state_dict",
    ):
        value = payload.get(key)

        if isinstance(value, dict):
            return value

    return payload


def _strip_uniform_prefix(
    state_dict: dict,
    prefix: str,
) -> dict:
    keys = list(state_dict)

    if keys and all(key.startswith(prefix) for key in keys):
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def _safe_torch_load(
    checkpoint_path: Path,
    device: torch.device,
):
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def load_trained_model(
    spec: ModelSpec,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    model = spec.model_class(deepcopy(spec.config)).to(device)

    payload = _safe_torch_load(
        checkpoint_path,
        device,
    )
    state_dict = _extract_state_dict(payload)

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Unsupported checkpoint format: "
            f"{checkpoint_path}"
        )

    state_dict = _strip_uniform_prefix(
        state_dict,
        "module.",
    )
    state_dict = _strip_uniform_prefix(
        state_dict,
        "_orig_mod.",
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint is incompatible with the current full-retraining "
            f"architecture '{spec.model_name}': {checkpoint_path}. "
            "Historical architecture overrides are intentionally disabled. "
            f"Original error: {error}"
        ) from error

    model.eval()
    return model