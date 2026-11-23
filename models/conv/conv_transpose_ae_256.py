from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAe256(ConvTransposeAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ConvTransposeAe256
config_path = "configs/conv_transpose_ae_256.json"
