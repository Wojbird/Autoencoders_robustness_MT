from models.residual.residual_ae_base import ResidualAEBase


class ResidualAe256(ResidualAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = ResidualAe256
config_path = "configs/residual/residual_ae_256.json"
