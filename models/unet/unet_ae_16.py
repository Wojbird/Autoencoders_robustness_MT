from models.unet.unet_ae_base import UNetAEBase


class UnetAe16(UNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = UnetAe16
config_path = "configs/unet/unet_ae_16.json"
