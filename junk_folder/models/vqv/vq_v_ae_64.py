from models.vqv.vq_v_ae_base import VQVAEBase


class VQVAE64(VQVAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = VQVAE64
config_path = "configs/vqv/vq_v_ae_64.json"