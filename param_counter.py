import importlib
import json


MODELS = [
    "conv.conv_transpose_ae_64",
    "conv.conv_transpose_ae_128",
    "conv.conv_transpose_ae_256",
    "conv.conv_transpose_ae_512",

    "residual.residual_ae_64",
    "residual.residual_ae_128",
    "residual.residual_ae_256",
    "residual.residual_ae_512",

    "unet.unet_ae_64",
    "unet.unet_ae_128",
    "unet.unet_ae_256",
    "unet.unet_ae_512",

    "adversarial.adversarial_ae_64",
    "adversarial.adversarial_ae_128",
    "adversarial.adversarial_ae_256",
    "adversarial.adversarial_ae_512",

    "vqv.vq_v_ae_64",
    "vqv.vq_v_ae_128",
    "vqv.vq_v_ae_256",
    "vqv.vq_v_ae_512",
]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compression_ratio(image_size, image_channels, latent_channels):
    input_size = image_size * image_size * image_channels
    latent_size = latent_channels * (image_size // 32) * (image_size // 32)
    return input_size / latent_size


print("=" * 120)

for model_path in MODELS:
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

    print(
        f"{cfg['name']:32} | "
        f"params: {total_params:>12,} | "
        f"latent: {latent_channels:>4} x {latent_hw} x {latent_hw} "
        f"= {latent_size:>6,} | "
        f"compression: {ratio:>6.2f}:1"
    )

print("=" * 120)