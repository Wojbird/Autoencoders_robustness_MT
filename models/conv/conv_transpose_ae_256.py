from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAE256(ConvTransposeAEBase):
    def __init__(self, config: dict):
        super().__init__(
            image_channels=config["image_channels"],
            ch1=64,
            ch2=128,
            ch3=192,
            ch4=224,
            latent_channels=256,
            dropout=0.2,
        )


model_class = ConvTransposeAE256
config_path = "configs/conv/conv_transpose_ae_256.json"