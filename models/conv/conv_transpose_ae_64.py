from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAE64(ConvTransposeAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = ConvTransposeAE64
config_path = "configs/conv/conv_transpose_ae_64.json"