from models.unet.unet_ae_base import UNetAEBase


class UNetAE256(UNetAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = UNetAE256
config_path = "configs/unet/unet_ae_256.json"