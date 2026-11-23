from models.vqv.vq_v_ae_base import VQVAEBase


class VqVAe16(VQVAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = VqVAe16
config_path = "configs/vq_v_ae_16.json"
