from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAe128(ConvTransposeAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ConvTransposeAe128
config_path = "configs/conv/conv_transpose_ae_128.json"
