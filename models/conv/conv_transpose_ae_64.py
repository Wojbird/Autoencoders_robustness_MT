from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAe64(ConvTransposeAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ConvTransposeAe64
config_path = "configs/conv_transpose_ae_64.json"
