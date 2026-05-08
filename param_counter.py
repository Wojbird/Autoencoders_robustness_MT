import importlib
import json


SIZES = [16, 32, 64, 128, 256, 512, 1024]

MODEL_PATTERNS = [
    "conv.conv_transpose_ae_{size}",
    "residual.residual_ae_{size}",
    "unet.unet_ae_{size}",
    "adversarial.adversarial_ae_{size}",
    "vqv.vq_v_ae_{size}",
]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compression_ratio(image_size, image_channels, latent_channels):
    input_size = image_size * image_size * image_channels
    latent_size = latent_channels * (image_size // 32) * (image_size // 32)
    return input_size / latent_size


print("=" * 130)

for size in SIZES:
    print(f"\nLATENT CHANNELS = {size}")
    print("-" * 130)

    rows = []

    for pattern in MODEL_PATTERNS:
        model_path = pattern.format(size=size)
        module = importlib.import_module(f"models.{model_path}")

        model_class = module.model_class
        config_path = module.config_path

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = model_class(cfg)

        total_params = count_params(model)

        latent_channels = int(cfg.get("latent_channels", cfg.get("latent_dim")))
        image_size = int(cfg["image_size"])
        image_channels = int(cfg["image_channels"])

        latent_hw = image_size // 32
        latent_size = latent_channels * latent_hw * latent_hw

        ratio = compression_ratio(
            image_size=image_size,
            image_channels=image_channels,
            latent_channels=latent_channels,
        )

        rows.append((cfg["name"], total_params, latent_channels, latent_hw, latent_size, ratio))

    max_params = max(r[1] for r in rows)

    for name, total_params, latent_channels, latent_hw, latent_size, ratio in rows:
        diff = 100.0 * (total_params - max_params) / max_params

        print(
            f"{name:32} | "
            f"params: {total_params:>12,} | "
            f"diff_vs_max: {diff:>7.2f}% | "
            f"latent: {latent_channels:>4} x {latent_hw} x {latent_hw} "
            f"= {latent_size:>6,} | "
            f"compression: {ratio:>7.2f}:1"
        )

print("\n" + "=" * 130)