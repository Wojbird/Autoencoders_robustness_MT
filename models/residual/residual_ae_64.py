from models.residual.residual_ae_base import ResidualAEBase


class ResidualAe64(ResidualAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ResidualAe64
config_path = "configs/residual_ae_64.json"
