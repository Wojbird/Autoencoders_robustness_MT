from models.unet_like.unet_like_ae_base import UNetLikeAEBase


class UNetLikeAE64(UNetLikeAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = UNetLikeAE64
config_path = "configs/unet_like/unet_like_ae_64.json"