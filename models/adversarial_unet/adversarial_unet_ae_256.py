from models.adversarial_unet.adversarial_unet_ae_base import AdversarialUNetAEBase


class AdversarialUnetAe256(AdversarialUNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = AdversarialUnetAe256
config_path = "configs/adversarial_unet_ae_256.json"
