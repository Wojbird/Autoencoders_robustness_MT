from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAE512(ConvTransposeAEBase):
    def __init__(self, config: dict):
        super().__init__(
            image_channels=config["image_channels"],
            ch1=128,
            ch2=256,
            ch3=384,
            ch4=448,
            latent_channels=512,
            dropout=0.2,
        )


model_class = ConvTransposeAE512
config_path = "configs/conv/conv_transpose_ae_512.json"