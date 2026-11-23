from models.adversarial_unet.adversarial_unet_ae_base import AdversarialUNetAEBase


class AdversarialUnetAe16(AdversarialUNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = AdversarialUnetAe16
config_path = "configs/adversarial_unet_ae_16.json"
