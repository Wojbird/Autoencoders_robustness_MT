from models.vqv.vq_v_ae_base import VQVAEBase


class VQVAE128(VQVAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = VQVAE128
config_path = "configs/vqv/vq_v_ae_128.json"
