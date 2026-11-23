from models.unet.unet_ae_base import UNetAEBase


class UnetAe32(UNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = UnetAe32
config_path = "configs/unet/unet_ae_32.json"
