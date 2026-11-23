from models.unet.unet_ae_base import UNetAEBase


class UnetAe64(UNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = UnetAe64
config_path = "configs/unet_ae_64.json"
