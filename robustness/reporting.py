from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_METRIC_COLUMNS = [
    "input_mse",
    "input_psnr",
    "input_ssim",
    "recon_mse",
    "recon_psnr",
    "recon_ssim",
    "fidelity_mse",
    "fidelity_psnr",
    "fidelity_ssim",
]

DERIVED_COLUMNS = [
    "baseline_recon_mse",
    "baseline_recon_psnr",
    "baseline_recon_ssim",
    "delta_mse",
    "delta_psnr",
    "delta_ssim",
    "relative_mse",
    "recovery_ratio",
]

SUMMARY_METRICS = BASE_METRIC_COLUMNS + DERIVED_COLUMNS

PLOT_METRICS = (
    "recon_mse",
    "recon_psnr",
    "recon_ssim",
    "relative_mse",
)


def _condition_key_from_file(
    path: Path,
) -> tuple[str, float]:
    frame = pd.read_csv(path, nrows=1)

    if frame.empty:
        raise ValueError(
            f"Empty raw result file: {path}"
        )

    return (
        str(frame.iloc[0]["attack"]),
        float(frame.iloc[0]["severity"]),
    )


def _read_and_collapse(
    paths: Iterable[Path],
) -> pd.DataFrame:
    """
    Combine files representing one logical condition.

    Rotation directions +theta and -theta are averaged for each original
    image, so they do not count as two independent observations.
    """
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    attack = str(frame["attack"].iloc[0])

    if attack != "rotation":
        return frame

    directions = {
        int(value)
        for value in frame["direction"].unique()
    }

    if directions != {-1, 1}:
        raise ValueError(
            "Rotation requires both directions -1 and 1; "
            f"found {sorted(directions)}."
        )

    metadata_columns = [
        "model_name",
        "family",
        "latent_channels",
        "latent_elements",
        "training_variant",
        "checkpoint_dataset",
        "evaluation_dataset",
        "checkpoint_path",
        "attack",
        "severity",
        "attack_space",
        "input_changed",
        "sample_index",
        "sample_path",
        "class_id",
    ]

    collapsed = (
        frame.groupby(
            metadata_columns,
            as_index=False,
            dropna=False,
        )[BASE_METRIC_COLUMNS]
        .mean()
    )

    collapsed["direction"] = 0

    return collapsed


def _derive_against_baseline(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    baseline_columns = (
        baseline[
            [
                "sample_index",
                "recon_mse",
                "recon_psnr",
                "recon_ssim",
            ]
        ]
        .rename(
            columns={
                "recon_mse": "baseline_recon_mse",
                "recon_psnr": "baseline_recon_psnr",
                "recon_ssim": "baseline_recon_ssim",
            }
        )
    )

    result = frame.merge(
        baseline_columns,
        on="sample_index",
        how="left",
        validate="many_to_one",
    )

    if result["baseline_recon_mse"].isna().any():
        raise ValueError(
            "Could not match every attacked row with its clean baseline row."
        )

    result["delta_mse"] = (
        result["recon_mse"]
        - result["baseline_recon_mse"]
    )

    result["delta_psnr"] = (
        result["recon_psnr"]
        - result["baseline_recon_psnr"]
    )

    result["delta_ssim"] = (
        result["recon_ssim"]
        - result["baseline_recon_ssim"]
    )

    recon_mse = pd.to_numeric(
        result["recon_mse"],
        errors="coerce",
    ).astype(float)

    baseline_mse = pd.to_numeric(
        result["baseline_recon_mse"],
        errors="coerce",
    ).astype(float)

    input_mse = pd.to_numeric(
        result["input_mse"],
        errors="coerce",
    ).astype(float)

    result["relative_mse"] = (
        recon_mse
        / baseline_mse.clip(lower=1e-12)
    )

    input_changed = result["input_changed"].astype(bool)

    recovery_ratio = pd.Series(
        np.nan,
        index=result.index,
        dtype=float,
    )

    recovery_ratio.loc[input_changed] = (
        recon_mse.loc[input_changed]
        / input_mse.loc[input_changed].clip(lower=1e-12)
    )

    result["recovery_ratio"] = recovery_ratio

    return result


def _summary_row(
    frame: pd.DataFrame,
) -> dict:
    first = frame.iloc[0]

    row = {
        "model_name": first["model_name"],
        "family": first["family"],
        "latent_channels": int(first["latent_channels"]),
        "latent_elements": int(first["latent_elements"]),
        "training_variant": first["training_variant"],
        "checkpoint_dataset": first["checkpoint_dataset"],
        "evaluation_dataset": first["evaluation_dataset"],
        "attack": first["attack"],
        "severity": float(first["severity"]),
        "n_images": int(frame["sample_index"].nunique()),
    }

    for metric in SUMMARY_METRICS:
        values = pd.to_numeric(
            frame[metric],
            errors="coerce",
        )

        finite_values = values[np.isfinite(values)]
        count = int(finite_values.shape[0])

        if count == 0:
            row[f"{metric}_mean"] = np.nan
            row[f"{metric}_std"] = np.nan
            row[f"{metric}_median"] = np.nan
            row[f"{metric}_ci95_low"] = np.nan
            row[f"{metric}_ci95_high"] = np.nan
            continue

        mean = float(finite_values.mean())

        std = (
            float(finite_values.std(ddof=1))
            if count > 1
            else 0.0
        )

        half_width = (
            1.96 * std / math.sqrt(count)
            if count > 1
            else 0.0
        )

        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_median"] = float(
            finite_values.median()
        )
        row[f"{metric}_ci95_low"] = mean - half_width
        row[f"{metric}_ci95_high"] = mean + half_width

    return row


def _complete_model_names(
    summary: pd.DataFrame,
    required_variants: set[str],
    *,
    allowed_families: set[str] | None = None,
) -> list[str]:
    frame = summary.copy()

    if allowed_families is not None:
        frame = frame[
            frame["family"].isin(allowed_families)
        ]

    baseline = frame[frame["attack"] == "clean"]

    variants_by_model = (
        baseline.groupby("model_name")["training_variant"]
        .agg(lambda values: set(values))
    )

    return sorted(
        model_name
        for model_name, variants in variants_by_model.items()
        if required_variants.issubset(variants)
    )


def _write_best_variant_table(
    summary: pd.DataFrame,
    summary_directory: Path,
) -> None:
    attacked = summary[summary["attack"] != "clean"].copy()

    if attacked.empty:
        return

    key_columns = [
        "model_name",
        "family",
        "latent_channels",
        "latent_elements",
        "attack",
        "severity",
    ]

    availability = (
        attacked.groupby(key_columns)["training_variant"]
        .agg(lambda values: sorted(set(values)))
        .reset_index(name="available_variants")
    )

    availability["available_variants"] = availability[
        "available_variants"
    ].apply(lambda values: ",".join(values))

    availability["n_variants_available"] = availability[
        "available_variants"
    ].apply(
        lambda value: 0 if not value else len(value.split(","))
    )

    best_indices = (
        attacked.groupby(key_columns)["recon_mse_mean"]
        .idxmin()
    )

    best = (
        attacked.loc[
            best_indices,
            key_columns
            + [
                "training_variant",
                "recon_mse_mean",
                "recon_psnr_mean",
                "recon_ssim_mean",
                "relative_mse_mean",
            ],
        ]
        .rename(
            columns={
                "training_variant": "best_training_variant_by_mse"
            }
        )
    )

    best = best.merge(
        availability,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    best["complete_three_way_comparison"] = (
        best["n_variants_available"] == 3
    )

    best.to_csv(
        summary_directory / "best_variant_by_condition.csv",
        index=False,
    )


def _write_coverage_table(
    summary: pd.DataFrame,
    summary_directory: Path,
) -> None:
    baseline = summary[summary["attack"] == "clean"]

    coverage = (
        baseline.groupby(
            [
                "model_name",
                "family",
                "latent_channels",
                "latent_elements",
            ],
            as_index=False,
        )["training_variant"]
        .agg(
            lambda values: ",".join(
                sorted(set(values))
            )
        )
        .rename(
            columns={
                "training_variant": "available_training_variants"
            }
        )
    )

    variant_sets = coverage[
        "available_training_variants"
    ].apply(
        lambda value: set(value.split(",")) if value else set()
    )

    coverage["has_clean"] = variant_sets.apply(
        lambda values: "clean" in values
    )

    coverage["has_noisy"] = variant_sets.apply(
        lambda values: "noisy" in values
    )

    coverage["has_noisy_latent"] = variant_sets.apply(
        lambda values: "noisy_latent" in values
    )

    coverage["complete_three_way"] = (
        coverage["has_clean"]
        & coverage["has_noisy"]
        & coverage["has_noisy_latent"]
    )

    coverage.to_csv(
        summary_directory / "checkpoint_coverage.csv",
        index=False,
    )


def _write_comparison_scope_table(
    summary: pd.DataFrame,
    summary_directory: Path,
) -> None:
    classical_models = _complete_model_names(
        summary,
        required_variants={
            "clean",
            "noisy",
            "noisy_latent",
        },
        allowed_families={
            "conv",
            "unet",
            "adversarial",
        },
    )

    clean_noisy_models = _complete_model_names(
        summary,
        required_variants={
            "clean",
            "noisy",
        },
    )

    rows = []

    for model_name in classical_models:
        rows.append(
            {
                "comparison": "three_way_classical",
                "model_name": model_name,
            }
        )

    for model_name in clean_noisy_models:
        rows.append(
            {
                "comparison": "clean_vs_noisy_all",
                "model_name": model_name,
            }
        )

    pd.DataFrame(rows).to_csv(
        summary_directory / "comparison_scope.csv",
        index=False,
    )


def _metric_label(metric: str) -> str:
    return {
        "recon_mse": "Reconstruction MSE",
        "recon_psnr": "Reconstruction PSNR [dB]",
        "recon_ssim": "Reconstruction SSIM",
        "relative_mse": "Relative MSE vs clean baseline",
    }[metric]


def _draw_curve_panel(
    axis,
    frame: pd.DataFrame,
    *,
    metric: str,
    show_confidence: bool,
) -> None:
    value_column = f"{metric}_mean"
    low_column = f"{metric}_ci95_low"
    high_column = f"{metric}_ci95_high"

    for variant in (
        "clean",
        "noisy",
        "noisy_latent",
    ):
        variant_frame = (
            frame[
                frame["training_variant"] == variant
            ]
            .sort_values("severity")
        )

        if variant_frame.empty:
            continue

        x = variant_frame["severity"].to_numpy(dtype=float)
        y = variant_frame[value_column].to_numpy(dtype=float)

        axis.plot(
            x,
            y,
            marker="o",
            label=variant,
        )

        if (
            show_confidence
            and low_column in variant_frame.columns
            and high_column in variant_frame.columns
        ):
            low = variant_frame[low_column].to_numpy(dtype=float)
            high = variant_frame[high_column].to_numpy(dtype=float)

            if np.isfinite(low).all() and np.isfinite(high).all():
                axis.fill_between(
                    x,
                    low,
                    high,
                    alpha=0.15,
                )

    axis.set_xlabel("Attack severity")
    axis.set_ylabel(_metric_label(metric))
    axis.grid(True, alpha=0.25)


def _plot_metric_grid(
    frame: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    show_confidence: bool,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        squeeze=False,
    )

    for axis, metric in zip(axes.flat, PLOT_METRICS):
        _draw_curve_panel(
            axis,
            frame,
            metric=metric,
            show_confidence=show_confidence,
        )

        axis.set_title(_metric_label(metric))

    handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            title="Training variant",
        )

    figure.suptitle(title, y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def _aggregate_matched_models(
    summary: pd.DataFrame,
    *,
    model_names: list[str],
    variants: set[str],
    output_csv: Path,
) -> pd.DataFrame:
    matched = summary[
        summary["model_name"].isin(model_names)
        & summary["training_variant"].isin(variants)
    ].copy()

    mean_columns = [
        column
        for column in summary.columns
        if column.endswith("_mean")
    ]

    aggregate = (
        matched.groupby(
            [
                "training_variant",
                "attack",
                "severity",
            ],
            as_index=False,
        )[mean_columns]
        .mean(numeric_only=True)
    )

    aggregate["n_model_configurations"] = len(model_names)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_csv, index=False)

    return aggregate


def _plot_aggregate(
    aggregate: pd.DataFrame,
    *,
    title_prefix: str,
    output_directory: Path,
) -> None:
    baseline = aggregate[aggregate["attack"] == "clean"]
    attacked = aggregate[aggregate["attack"] != "clean"]

    for attack, attack_frame in attacked.groupby("attack"):
        baseline_for_plot = baseline.copy()
        baseline_for_plot["attack"] = attack
        baseline_for_plot["severity"] = 0.0

        plot_frame = pd.concat(
            [
                baseline_for_plot,
                attack_frame,
            ],
            ignore_index=True,
        )

        _plot_metric_grid(
            plot_frame,
            title=f"{title_prefix}: {attack}",
            output_path=output_directory / f"{attack}.png",
            show_confidence=False,
        )


def generate_plots(
    summary: pd.DataFrame,
    output_root: Path,
) -> None:
    plots_root = output_root / "plots"
    baseline = summary[summary["attack"] == "clean"].copy()
    attacked = summary[summary["attack"] != "clean"].copy()

    for model_name, model_frame in attacked.groupby("model_name"):
        baseline_model = baseline[
            baseline["model_name"] == model_name
        ]

        for attack, attack_frame in model_frame.groupby("attack"):
            baseline_for_plot = baseline_model.copy()
            baseline_for_plot["attack"] = attack
            baseline_for_plot["severity"] = 0.0

            plot_frame = pd.concat(
                [
                    baseline_for_plot,
                    attack_frame,
                ],
                ignore_index=True,
            )

            _plot_metric_grid(
                plot_frame,
                title=f"{model_name}: {attack}",
                output_path=(
                    plots_root
                    / "by_model"
                    / model_name
                    / f"{attack}.png"
                ),
                show_confidence=True,
            )

    classical_models = _complete_model_names(
        summary,
        required_variants={
            "clean",
            "noisy",
            "noisy_latent",
        },
        allowed_families={
            "conv",
            "unet",
            "adversarial",
        },
    )

    classical_aggregate = _aggregate_matched_models(
        summary,
        model_names=classical_models,
        variants={
            "clean",
            "noisy",
            "noisy_latent",
        },
        output_csv=(
            output_root
            / "summary"
            / "aggregate_matched_classical.csv"
        ),
    )

    _plot_aggregate(
        classical_aggregate,
        title_prefix=(
            "Matched classical models "
            f"(n={len(classical_models)})"
        ),
        output_directory=(
            plots_root / "aggregate_matched_classical"
        ),
    )

    clean_noisy_models = _complete_model_names(
        summary,
        required_variants={
            "clean",
            "noisy",
        },
    )

    clean_noisy_aggregate = _aggregate_matched_models(
        summary,
        model_names=clean_noisy_models,
        variants={
            "clean",
            "noisy",
        },
        output_csv=(
            output_root
            / "summary"
            / "aggregate_clean_vs_noisy.csv"
        ),
    )

    _plot_aggregate(
        clean_noisy_aggregate,
        title_prefix=(
            "Clean vs noisy, matched models "
            f"(n={len(clean_noisy_models)})"
        ),
        output_directory=(
            plots_root / "aggregate_clean_vs_noisy"
        ),
    )

    vqv_summary = summary[summary["family"] == "vqv"]

    vqv_models = _complete_model_names(
        vqv_summary,
        required_variants={
            "clean",
            "noisy",
        },
    )

    vqv_aggregate = _aggregate_matched_models(
        summary,
        model_names=vqv_models,
        variants={
            "clean",
            "noisy",
        },
        output_csv=(
            output_root
            / "summary"
            / "aggregate_vqv.csv"
        ),
    )

    _plot_aggregate(
        vqv_aggregate,
        title_prefix=f"VQ-VAE models (n={len(vqv_models)})",
        output_directory=plots_root / "aggregate_vqv",
    )


def build_reports(
    output_root: Path,
    *,
    overwrite_derived: bool = True,
) -> Path:
    raw_root = output_root / "raw"

    if not raw_root.exists():
        raise FileNotFoundError(
            "Raw result directory does not exist: "
            f"{raw_root}"
        )

    summary_rows: list[dict] = []

    training_directories = sorted(
        path
        for path in raw_root.glob("*/*")
        if path.is_dir()
    )

    for training_directory in training_directories:
        raw_paths = sorted(training_directory.glob("*.csv"))

        if not raw_paths:
            continue

        baseline_paths = [
            path
            for path in raw_paths
            if path.name.startswith("clean__")
        ]

        if len(baseline_paths) != 1:
            raise ValueError(
                "Expected exactly one clean baseline in "
                f"{training_directory}, found {len(baseline_paths)}."
            )

        baseline = pd.read_csv(baseline_paths[0])
        grouped_paths: dict[tuple[str, float], list[Path]] = {}

        for path in raw_paths:
            grouped_paths.setdefault(
                _condition_key_from_file(path),
                [],
            ).append(path)

        relative_directory = training_directory.relative_to(raw_root)

        derived_directory = (
            output_root / "derived" / relative_directory
        )
        derived_directory.mkdir(parents=True, exist_ok=True)

        for (attack, severity), paths in sorted(grouped_paths.items()):
            frame = _read_and_collapse(paths)

            derived = _derive_against_baseline(
                frame,
                baseline,
            )

            severity_text = (
                f"{severity:.10g}"
                .replace(".", "p")
                .replace("-", "m")
            )

            derived_path = (
                derived_directory
                / f"{attack}__{severity_text}.csv"
            )

            if overwrite_derived or not derived_path.exists():
                temporary_path = derived_path.with_suffix(
                    derived_path.suffix + ".tmp"
                )

                derived.to_csv(
                    temporary_path,
                    index=False,
                )

                temporary_path.replace(derived_path)

            summary_rows.append(_summary_row(derived))

    if not summary_rows:
        raise ValueError(
            "No raw result files were found."
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "family",
                "latent_channels",
                "model_name",
                "training_variant",
                "attack",
                "severity",
            ]
        )
    )

    summary_directory = output_root / "summary"
    summary_directory.mkdir(parents=True, exist_ok=True)

    summary_path = summary_directory / "summary_metrics.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    _write_best_variant_table(summary, summary_directory)
    _write_coverage_table(summary, summary_directory)
    _write_comparison_scope_table(summary, summary_directory)
    generate_plots(summary, output_root)

    return summary_path