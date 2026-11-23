from models.adversarial_unet.adversarial_unet_ae_base import AdversarialUNetAEBase


class AdversarialUnetAe128(AdversarialUNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = AdversarialUnetAe128
config_path = "configs/adversarial_unet_ae_128.json"
