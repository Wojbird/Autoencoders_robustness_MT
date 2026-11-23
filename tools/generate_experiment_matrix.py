from __future__ import annotations

import argparse
import json
from pathlib import Path


LATENT_CHANNELS = (16, 32, 64, 128, 256)

FAMILIES = {
    "conv_transpose_ae": {
        "registry_family": "conv",
        "module_dir": "conv",
        "base_module": "models.conv.conv_transpose_ae_base",
        "base_class": "ConvTransposeAEBase",
        "file_prefix": "conv_transpose_ae",
        "channels": {
            16: [28, 40, 58, 78, 102],
            32: [30, 42, 58, 80, 104],
            64: [30, 44, 60, 82, 108],
            128: [38, 54, 76, 104, 136],
            256: [46, 66, 92, 126, 166],
        },
    },
    "residual_ae": {
        "registry_family": "residual",
        "module_dir": "residual",
        "base_module": "models.residual.residual_ae_base",
        "base_class": "ResidualAEBase",
        "file_prefix": "residual_ae",
        "stem_channels": {
            16: [8, 12],
            32: [8, 12],
            64: [8, 12],
            128: [10, 16],
            256: [14, 20],
        },
        "channels": {
            16: [18, 26, 34, 42, 52],
            32: [18, 28, 36, 44, 52],
            64: [20, 28, 36, 46, 54],
            128: [24, 36, 46, 58, 68],
            256: [30, 44, 58, 70, 84],
        },
    },
    "u_shaped_ae": {
        "registry_family": "u_shaped",
        "module_dir": "u_shaped",
        "base_module": "models.u_shaped.u_shaped_ae_base",
        "base_class": "UShapedAEBase",
        "file_prefix": "u_shaped_ae",
        "channels": {
            16: [14, 20, 30, 42, 54],
            32: [14, 20, 30, 42, 56],
            64: [16, 22, 32, 44, 58],
            128: [20, 28, 40, 56, 72],
            256: [24, 34, 48, 70, 90],
        },
    },
    "unet_ae": {
        "registry_family": "unet",
        "module_dir": "unet",
        "base_module": "models.unet.unet_ae_base",
        "base_class": "UNetAEBase",
        "file_prefix": "unet_ae",
        "channels": {
            16: [14, 20, 28, 38, 50],
            32: [14, 20, 28, 38, 50],
            64: [14, 20, 28, 40, 52],
            128: [18, 26, 36, 52, 68],
            256: [22, 32, 44, 64, 82],
        },
    },
    "adversarial_unet_ae": {
        "registry_family": "adversarial_unet",
        "module_dir": "adversarial_unet",
        "base_module": "models.adversarial_unet.adversarial_unet_ae_base",
        "base_class": "AdversarialUNetAEBase",
        "file_prefix": "adversarial_unet_ae",
        "channels": {
            16: [14, 20, 28, 38, 50],
            32: [14, 20, 28, 38, 50],
            64: [14, 20, 28, 40, 52],
            128: [18, 26, 36, 52, 68],
            256: [22, 32, 44, 64, 82],
        },
    },
    "vq_v_ae": {
        "registry_family": "vqv",
        "module_dir": "vqv",
        "base_module": "models.vqv.vq_v_ae_base",
        "base_class": "VQVAEBase",
        "file_prefix": "vq_v_ae",
        "stem_channels": {
            16: [14, 22],
            32: [14, 22],
            64: [14, 22],
            128: [18, 28],
            256: [22, 34],
        },
        "channels": {
            16: [32, 44, 58, 72],
            32: [32, 44, 58, 74],
            64: [32, 44, 58, 74],
            128: [40, 56, 74, 92],
            256: [46, 68, 88, 110],
        },
        "bottleneck_channels": {16: 88, 32: 90, 64: 90, 128: 112, 256: 134},
    },
}


def common_config(family: str, latent_channels: int) -> dict:
    spec = FAMILIES[family]
    return {
        "name": f"{family}_{latent_channels}",
        "family": spec["registry_family"],
        "image_size": 224,
        "image_channels": 3,
        "latent_channels": latent_channels,
        "latent_spatial_size": 7,
        "latent_size": latent_channels * 7 * 7,
        "dropout": 0.2,
        "batch_size": 32,
        "num_workers": 8,
        "epochs": 150,
        "learning_rate": 0.00005,
        "weight_decay": 0.00005,
        "noise_std": 0.1,
        "noise_latent": 0.1,
        "latent_noise_relative": True,
        "eval_noise_std": 0.1,
        "train_eval_fraction": 0.1,
        "val_subset_fraction": 1.0,
        "early_stopping_patience": 5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "scheduler_min_lr": 0.000001,
        "scheduler_threshold": 0.0001,
        "experiment_seed": 42,
        "dataset_split_seed": 42,
        "dataloader_seed": 42,
        "eval_noise_seed": 1234,
    }


def build_config(family: str, latent_channels: int) -> dict:
    spec = FAMILIES[family]
    config = common_config(family, latent_channels)
    config["encoder_channels"] = spec["channels"][latent_channels]

    if "stem_channels" in spec:
        config["stem_channels"] = spec["stem_channels"][latent_channels]

    if family == "vq_v_ae":
        config.update(
            {
                "latent_dim": latent_channels,
                "num_embeddings": 512,
                "commitment_cost": 0.25,
                "bottleneck_channels": spec["bottleneck_channels"][latent_channels],
            }
        )
    elif family == "adversarial_unet_ae":
        config.update(
            {
                "disc_channels": [32, 64, 96, 128],
                "disc_learning_rate": 0.0001,
                "disc_weight_decay": 0.00001,
                "adversarial_weight": 0.001,
            }
        )

    return config


def wrapper_source(family: str, latent_channels: int) -> str:
    spec = FAMILIES[family]
    class_name = "".join(part.capitalize() for part in family.split("_")) + str(latent_channels)
    return (
        f'from {spec["base_module"]} import {spec["base_class"]}\n\n\n'
        f'class {class_name}({spec["base_class"]}):\n'
        '    """Configuration-bound model wrapper."""\n\n'
        '    pass\n\n\n'
        f'model_class = {class_name}\n'
        f'config_path = "configs/{family}_{latent_channels}.json"\n'
    )


def generate(output_root: Path) -> None:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for family, spec in FAMILIES.items():
        model_dir = output_root / "models" / spec["module_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)

        for latent_channels in LATENT_CHANNELS:
            config_path = config_dir / f"{family}_{latent_channels}.json"
            config_path.write_text(
                json.dumps(build_config(family, latent_channels), indent=2) + "\n",
                encoding="utf-8",
            )

            wrapper_path = model_dir / f'{spec["file_prefix"]}_{latent_channels}.py'
            wrapper_path.write_text(
                wrapper_source(family, latent_channels),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the canonical 6 x 5 experiment matrix.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project overlay root (defaults to the parent of tools/).",
    )
    args = parser.parse_args()
    generate(args.output_root.resolve())


if __name__ == "__main__":
    main()
