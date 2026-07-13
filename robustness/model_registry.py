from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Iterable

import torch


MAIN_EXPERIMENT_SIZES = {
    16,
    32,
    64,
    128,
    256,
}

FAMILY_ALIASES = {
    "conv": "conv",
    "convolutional": "conv",
    "unet": "unet",
    "adversarial": "adversarial",
    "adv": "adversarial",
    "vqv": "vqv",
    "vq": "vqv",
    "vqvae": "vqv",
}

OVERRIDES_RELATIVE_PATH = Path(
    "configs/checkpoint_architecture_overrides.json"
)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    family: str
    latent_channels: int
    latent_elements: int
    module_name: str
    module_path: Path
    config_path: Path
    model_class: type
    config: dict
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
            / (
                f"{self.model_name}_"
                f"{training_variant}_best.pt"
            )
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
        return bool(self.config_override)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_checkpoint_overrides(
    project_root: Path,
) -> dict[str, dict]:
    path = project_root / OVERRIDES_RELATIVE_PATH

    if not path.exists():
        return {}

    payload = _load_json(path)
    models = payload.get("models", {})

    if not isinstance(models, dict):
        raise TypeError(
            f"'{path}' must contain a JSON object in the 'models' field."
        )

    result: dict[str, dict] = {}

    for model_name, override in models.items():
        if not isinstance(override, dict):
            raise TypeError(
                f"Override for '{model_name}' must be a JSON object."
            )

        result[str(model_name)] = deepcopy(override)

    return result


def _merge_config(
    base_config: dict,
    override: dict,
) -> dict:
    effective_config = deepcopy(base_config)

    for key, value in override.items():
        effective_config[key] = deepcopy(value)

    return effective_config


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


def discover_model_specs(
    project_root: Path,
    selectors: Iterable[str] | None = None,
) -> list[ModelSpec]:
    project_root = project_root.resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    checkpoint_overrides = _load_checkpoint_overrides(project_root)
    models_root = project_root / "models"
    specs: list[ModelSpec] = []

    for family in (
        "conv",
        "unet",
        "adversarial",
        "vqv",
    ):
        family_root = models_root / family

        for module_path in sorted(family_root.glob("*.py")):
            if (
                module_path.name.endswith("_base.py")
                or module_path.name == "__init__.py"
            ):
                continue

            suffix = module_path.stem.rsplit("_", maxsplit=1)[-1]

            if (
                not suffix.isdigit()
                or int(suffix) not in MAIN_EXPERIMENT_SIZES
            ):
                continue

            relative_module_path = (
                module_path
                .with_suffix("")
                .relative_to(project_root)
            )

            module_name = ".".join(relative_module_path.parts)
            module = importlib.import_module(module_name)

            model_class = getattr(module, "model_class", None)
            config_value = getattr(module, "config_path", None)

            if model_class is None or config_value is None:
                raise AttributeError(
                    f"{module_name} must define model_class and config_path."
                )

            config_path = (project_root / config_value).resolve()
            base_config = _load_json(config_path)
            model_name = str(base_config["name"])
            config_override = checkpoint_overrides.get(model_name, {})

            effective_config = _merge_config(
                base_config,
                config_override,
            )

            config_family = str(
                effective_config.get("family", family)
            ).lower()

            latent_channels = int(
                effective_config.get(
                    "latent_channels",
                    effective_config.get(
                        "latent_dim",
                        int(suffix),
                    ),
                )
            )

            latent_spatial_size = int(
                effective_config.get(
                    "latent_spatial_size",
                    7,
                )
            )

            latent_elements = int(
                effective_config.get(
                    "latent_size",
                    (
                        latent_channels
                        * latent_spatial_size
                        * latent_spatial_size
                    ),
                )
            )

            specs.append(
                ModelSpec(
                    model_name=model_name,
                    family=config_family,
                    latent_channels=latent_channels,
                    latent_elements=latent_elements,
                    module_name=module_name,
                    module_path=module_path,
                    config_path=config_path,
                    model_class=model_class,
                    config=effective_config,
                    config_override=deepcopy(config_override),
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
            family_selector = FAMILY_ALIASES.get(selector)
            exact_model_match = selector == spec.model_name.lower()
            family_match = family_selector == spec.family

            if exact_model_match or family_match:
                selected.append(spec)
                break

    if not selected:
        available = sorted(spec.model_name for spec in specs)

        raise ValueError(
            "No model matched selectors "
            f"{normalised_selectors}. Available models: {available}"
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
    model = spec.model_class(spec.config).to(device)

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
        override_note = (
            "Applied checkpoint override: "
            f"{spec.config_override}."
            if spec.uses_checkpoint_override
            else "No checkpoint architecture override was applied."
        )

        raise RuntimeError(
            "Checkpoint is incompatible with model "
            f"'{spec.model_name}': {checkpoint_path}. "
            f"{override_note} Original error: {error}"
        ) from error

    model.eval()

    return model