from __future__ import annotations

from pathlib import Path

import pytest
import torch

from robustness.model_registry import discover_model_specs
from robustness.attacks import SUPPORTED_ATTACKS, build_attack_conditions
from utils.latent import forward_with_latent_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_AUTOENCODER_PARAMETERS = {
    ("conv", 16): 292_595,
    ("conv", 32): 309_407,
    ("conv", 64): 336_119,
    ("conv", 128): 546_431,
    ("conv", 256): 839_063,
    ("residual", 16): 291_467,
    ("residual", 32): 315_915,
    ("residual", 64): 338_923,
    ("residual", 128): 547_567,
    ("residual", 256): 845_855,
    ("u_shaped", 16): 291_995,
    ("u_shaped", 32): 305_783,
    ("u_shaped", 64): 339_703,
    ("u_shaped", 128): 538_683,
    ("u_shaped", 256): 845_559,
    ("unet", 16): 299_671,
    ("unet", 32): 301_303,
    ("unet", 64): 325_383,
    ("unet", 128): 556_963,
    ("unet", 256): 839_323,
    ("adversarial_unet", 16): 299_671,
    ("adversarial_unet", 32): 301_303,
    ("adversarial_unet", 64): 325_383,
    ("adversarial_unet", 128): 556_963,
    ("adversarial_unet", 256): 839_323,
    ("vqv", 16): 289_143,
    ("vqv", 32): 308_229,
    ("vqv", 64): 330_405,
    ("vqv", 128): 544_737,
    ("vqv", 256): 841_741,
}


def _count_trainable(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


@pytest.fixture(scope="module")
def specs():
    return discover_model_specs(PROJECT_ROOT, selectors=["all"])


def test_registry_discovers_exactly_the_canonical_6_by_5_matrix(specs):
    assert len(specs) == 30
    assert {spec.family for spec in specs} == {
        "conv",
        "residual",
        "u_shaped",
        "unet",
        "adversarial_unet",
        "vqv",
    }
    assert {spec.latent_channels for spec in specs} == {16, 32, 64, 128, 256}
    assert len({spec.model_name for spec in specs}) == 30


def test_declared_parameter_budgets_are_exact(specs):
    observed = {}
    for spec in specs:
        model = spec.model_class(spec.config)
        observed[(spec.family, spec.latent_channels)] = _count_trainable(model)

    assert observed == EXPECTED_AUTOENCODER_PARAMETERS


def test_parameter_spread_is_at_most_five_percent(specs):
    for latent_channels in (16, 32, 64, 128, 256):
        counts = [
            EXPECTED_AUTOENCODER_PARAMETERS[(spec.family, latent_channels)]
            for spec in specs
            if spec.latent_channels == latent_channels
        ]
        spread_pct = 100.0 * (max(counts) - min(counts)) / min(counts)
        assert spread_pct <= 5.0


@pytest.mark.parametrize(
    "family",
    ["conv", "residual", "u_shaped", "unet", "adversarial_unet", "vqv"],
)
def test_size_16_forward_shape_and_range(specs, family):
    spec = next(
        item
        for item in specs
        if item.family == family and item.latent_channels == 16
    )
    model = spec.model_class(spec.config).eval()
    x = torch.rand(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


@pytest.mark.parametrize("family", ["unet", "adversarial_unet"])
def test_true_unet_supports_shape_preserving_latent_transform(specs, family):
    spec = next(
        item
        for item in specs
        if item.family == family and item.latent_channels == 16
    )
    model = spec.model_class(spec.config).eval()
    x = torch.rand(1, 3, 224, 224)
    seen_shapes = []

    def transform(z: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(tuple(z.shape))
        return z

    with torch.no_grad():
        output = forward_with_latent_transform(model, x, transform)

    assert seen_shapes == [(1, 16, 7, 7)]
    assert output.shape == x.shape


def test_adversarial_discriminator_is_reported_separately(specs):
    spec = next(
        item
        for item in specs
        if item.family == "adversarial_unet" and item.latent_channels == 16
    )
    generator = spec.model_class(spec.config)
    discriminator = generator.discriminator_class(spec.config)

    assert _count_trainable(generator) == 299_671
    assert _count_trainable(discriminator) == 185_889


def test_vq_models_do_not_advertise_latent_noise_training(specs):
    for spec in specs:
        if spec.family == "vqv":
            assert spec.supported_training_variants == ("clean", "noisy")
            assert spec.supports_latent_attack is False
        else:
            assert spec.supports_latent_attack is True


def test_robustness_protocol_excludes_rotation():
    assert SUPPORTED_ATTACKS == (
        "gaussian_input",
        "gaussian_latent",
        "impulse",
        "occlusion",
    )

    config = {
        "attacks": {
            attack: {"levels": [0.1]}
            for attack in SUPPORTED_ATTACKS
        }
    }
    conditions = build_attack_conditions(config, selected=["all"])
    assert [condition.name for condition in conditions] == [
        "clean",
        "gaussian_input",
        "gaussian_latent",
        "impulse",
        "occlusion",
    ]
