from models.vqv.vq_v_ae_base import VQVAEBase


class VqVAe256(VQVAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = VqVAe256
config_path = "configs/vq_v_ae_256.json"
