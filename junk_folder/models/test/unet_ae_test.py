from models.unet.unet_ae_base import UNetAEBase


class UNetAETest(UNetAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = UNetAETest
config_path = "configs/test/unet_ae_test.json"