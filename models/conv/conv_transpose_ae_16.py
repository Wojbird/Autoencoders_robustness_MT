from models.conv.conv_transpose_ae_base import ConvTransposeAEBase


class ConvTransposeAe16(ConvTransposeAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ConvTransposeAe16
config_path = "configs/conv/conv_transpose_ae_16.json"
