from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAE1024(ConvTransposeAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = ConvTransposeAE1024
config_path = "configs/conv/conv_transpose_ae_1024.json"