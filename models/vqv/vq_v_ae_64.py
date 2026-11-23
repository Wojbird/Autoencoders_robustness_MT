from models.vqv.vq_v_ae_base import VQVAEBase


class VqVAe64(VQVAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = VqVAe64
config_path = "configs/vqv/vq_v_ae_64.json"
