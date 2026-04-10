from models.vqv.vq_v_ae_base import VQVAEBase


class VQVAE1024(VQVAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = VQVAE1024
config_path = "configs/vqv/vq_v_ae_1024.json"