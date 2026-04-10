from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAETest(ConvTransposeAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = ConvTransposeAETest
config_path = "configs/test/conv_transpose_ae_test.json"