from models.u_shaped.u_shaped_ae_base import UNetAEBase


class UNetAE256(UNetAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = UNetAE256
config_path = "configs/u_shaped/unet_ae_256.json"