from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAe32(ConvTransposeAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ConvTransposeAe32
config_path = "configs/conv/conv_transpose_ae_32.json"
