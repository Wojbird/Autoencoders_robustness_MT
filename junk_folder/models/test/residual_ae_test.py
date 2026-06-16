from models.residual.residual_ae_base import ResidualAEBase


class ResidualAETest(ResidualAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = ResidualAETest
config_path = "configs/test/residual_ae_test.json"