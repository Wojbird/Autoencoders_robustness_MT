from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAE128(ConvTransposeAEBase):
    def __init__(self, config: dict):
        super().__init__(
            image_channels=config["image_channels"],
            ch1=32,
            ch2=64,
            ch3=96,
            ch4=112,
            latent_channels=128,
            dropout=0.2,
        )


model_class = ConvTransposeAE128
config_path = "configs/conv/conv_transpose_ae_128.json"