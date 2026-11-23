from models.residual.residual_ae_base import ResidualAEBase


class ResidualAe32(ResidualAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ResidualAe32
config_path = "configs/residual/residual_ae_32.json"
