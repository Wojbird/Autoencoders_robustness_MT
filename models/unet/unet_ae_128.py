from models.unet.unet_ae_base import UNetAEBase


class UnetAe128(UNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = UnetAe128
config_path = "configs/unet_ae_128.json"
