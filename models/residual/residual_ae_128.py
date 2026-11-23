from models.residual.residual_ae_base import ResidualAEBase


class ResidualAe128(ResidualAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ResidualAe128
config_path = "configs/residual_ae_128.json"
