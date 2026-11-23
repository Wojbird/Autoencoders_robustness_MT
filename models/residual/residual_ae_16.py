from models.residual.residual_ae_base import ResidualAEBase


class ResidualAe16(ResidualAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ResidualAe16
config_path = "configs/residual_ae_16.json"
