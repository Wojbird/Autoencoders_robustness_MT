from models.unet_like.unet_like_ae_base import UNetLikeAEBase


class UNetLikeAE128(UNetLikeAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = UNetLikeAE128
config_path = "configs/unet_like/unet_like_ae_128.json"