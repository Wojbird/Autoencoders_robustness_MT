from models.unet.unet_ae_base import UNetAEBase


class UNetAE1024(UNetAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = UNetAE1024
config_path = "configs/unet/unet_ae_1024.json"