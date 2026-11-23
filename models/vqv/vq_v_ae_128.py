from models.vqv.vq_v_ae_base import VQVAEBase


class VqVAe128(VQVAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = VqVAe128
config_path = "configs/vqv/vq_v_ae_128.json"
