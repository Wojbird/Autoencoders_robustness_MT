from models.adversarial.adversarial_unet_like_ae_base import AdversarialUNetAEBase


class AdversarialUNetAE256(AdversarialUNetAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialUNetAE256
config_path = "configs/adversarial/adversarial_unet_like_ae_256.json"