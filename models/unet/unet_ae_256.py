from models.unet.unet_ae_base import UNetAEBase


class UnetAe256(UNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = UnetAe256
config_path = "configs/unet/unet_ae_256.json"
