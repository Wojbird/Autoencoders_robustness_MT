from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAE64(ConvTransposeAEBase):
    def __init__(self, config: dict):
        super().__init__(
            image_channels=config["image_channels"],
            ch1=16,
            ch2=32,
            ch3=48,
            ch4=56,
            latent_channels=64,
            dropout=0.2,
        )


model_class = ConvTransposeAE64
config_path = "configs/conv/conv_transpose_ae_64.json"