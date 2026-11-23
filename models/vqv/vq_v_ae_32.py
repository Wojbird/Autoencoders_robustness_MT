from models.vqv.vq_v_ae_base import VQVAEBase


class VqVAe32(VQVAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = VqVAe32
config_path = "configs/vqv/vq_v_ae_32.json"
