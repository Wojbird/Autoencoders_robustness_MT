from models.unet_like.unet_like_ae_base import UNetLikeAEBase


class UNetLikeAE1024(UNetLikeAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = UNetLikeAE1024
config_path = "configs/unet_like/unet_like_ae_1024.json"