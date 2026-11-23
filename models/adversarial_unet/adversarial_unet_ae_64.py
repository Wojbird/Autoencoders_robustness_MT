from models.adversarial_unet.adversarial_unet_ae_base import AdversarialUNetAEBase


class AdversarialUnetAe64(AdversarialUNetAEBase):
    """Configuration-bound model wrapper."""

    pass


model_class = AdversarialUnetAe64
config_path = "configs/adversarial_unet/adversarial_unet_ae_64.json"
