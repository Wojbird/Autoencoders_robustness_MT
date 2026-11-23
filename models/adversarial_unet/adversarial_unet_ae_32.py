from models.adversarial_unet.adversarial_unet_ae_base import AdversarialUNetAEBase


class AdversarialUnetAe32(AdversarialUNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = AdversarialUnetAe32
config_path = "configs/adversarial_unet/adversarial_unet_ae_32.json"
