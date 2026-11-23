from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from robustness.model_registry import discover_model_specs


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count and validate parameters for the canonical model matrix."
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Exact model name, registered family/alias, or 'all'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/model_parameter_counts.csv"),
    )
    parser.add_argument(
        "--max-spread-pct",
        type=float,
        default=5.0,
        help="Maximum permitted (max-min)/min spread within one latent size.",
    )
    return parser.parse_args()


def count_trainable_parameters(module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def count_all_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def build_row(spec) -> dict:
    model = spec.model_class(spec.config)
    autoencoder_trainable = count_trainable_parameters(model)
    autoencoder_all = count_all_parameters(model)

    discriminator_class = getattr(model, "discriminator_class", None)
    if callable(discriminator_class):
        discriminator = discriminator_class(spec.config)
        discriminator_trainable = count_trainable_parameters(discriminator)
        discriminator_all = count_all_parameters(discriminator)
    else:
        discriminator_trainable = 0
        discriminator_all = 0

    image_size = int(spec.config["image_size"])
    image_channels = int(spec.config["image_channels"])
    input_elements = image_size * image_size * image_channels
    central_compression_ratio = input_elements / spec.latent_elements
    has_skips = spec.family in {"unet", "adversarial_unet"}
    latent_type = "discrete_vq" if spec.family == "vqv" else "continuous"

    return {
        "model_name": spec.model_name,
        "family": spec.family,
        "latent_channels": spec.latent_channels,
        "latent_spatial_size": image_size // 32,
        "latent_elements": spec.latent_elements,
        "latent_type": latent_type,
        "has_skip_connections": has_skips,
        "autoencoder_trainable_params": autoencoder_trainable,
        "autoencoder_all_params": autoencoder_all,
        "discriminator_trainable_params": discriminator_trainable,
        "discriminator_all_params": discriminator_all,
        "system_trainable_params": autoencoder_trainable + discriminator_trainable,
        "central_tensor_compression_ratio": central_compression_ratio,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_spread_pct < 0.0:
        raise ValueError("--max-spread-pct must be non-negative.")

    os.chdir(PROJECT_ROOT)
    specs = discover_model_specs(PROJECT_ROOT, selectors=[args.model])
    if not specs:
        raise ValueError(f"No canonical model matched {args.model!r}.")

    rows = [build_row(spec) for spec in specs]
    rows.sort(key=lambda row: (row["latent_channels"], row["family"]))

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    write_csv(rows, output_path)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["latent_channels"])].append(row)

    failures: list[str] = []
    for latent_channels, group_rows in sorted(grouped.items()):
        counts = [int(row["autoencoder_trainable_params"]) for row in group_rows]
        minimum = min(counts)
        maximum = max(counts)
        spread_pct = 100.0 * (maximum - minimum) / minimum

        print(f"\nLATENT CHANNELS = {latent_channels} | spread={spread_pct:.2f}%")
        print("-" * 138)
        for row in group_rows:
            print(
                f"{row['model_name']:34} | "
                f"AE/G: {row['autoencoder_trainable_params']:>10,} | "
                f"D: {row['discriminator_trainable_params']:>9,} | "
                f"system: {row['system_trainable_params']:>10,} | "
                f"latent: {row['latent_channels']:>3}x"
                f"{row['latent_spatial_size']}x{row['latent_spatial_size']} | "
                f"central ratio: {row['central_tensor_compression_ratio']:>7.2f}:1"
            )

        if len(group_rows) > 1 and spread_pct > args.max_spread_pct:
            failures.append(
                f"C={latent_channels}: {spread_pct:.2f}% > {args.max_spread_pct:.2f}%"
            )

    print(f"\nCSV written to: {output_path}")
    print(
        "Note: central_tensor_compression_ratio is not an information-bottleneck "
        "ratio for U-Net models because skip connections bypass the central tensor."
    )
    print(
        "Note: VQ-VAE uses a discrete codebook, so its capacity is not fully "
        "described by the continuous tensor ratio."
    )

    if failures:
        raise RuntimeError(
            "Parameter matching threshold exceeded: " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
