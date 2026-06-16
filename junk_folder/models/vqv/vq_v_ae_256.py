from models.vqv.vq_v_ae_base import VQVAEBase


class VQVAE256(VQVAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = VQVAE256
config_path = "configs/vqv/vq_v_ae_256.json"