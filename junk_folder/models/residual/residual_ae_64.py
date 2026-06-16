from models.residual.residual_ae_base import ResidualAEBase


class ResidualAE64(ResidualAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = ResidualAE64
config_path = "configs/residual/residual_ae_64.json"