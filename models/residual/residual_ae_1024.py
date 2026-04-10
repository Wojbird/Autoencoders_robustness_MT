from models.residual.residual_ae_base import ResidualAEBase


class ResidualAE1024(ResidualAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = ResidualAE1024
config_path = "configs/residual/residual_ae_1024.json"