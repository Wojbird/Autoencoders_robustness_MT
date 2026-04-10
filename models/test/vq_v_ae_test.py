from models.vqv.vq_v_ae_base import VQVAEBase


class VQVAETest(VQVAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = VQVAETest
config_path = "configs/test/vq_v_ae_test.json"