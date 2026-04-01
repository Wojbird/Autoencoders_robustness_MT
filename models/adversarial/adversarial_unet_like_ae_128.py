from models.adversarial.adversarial_unet_like_ae_base import AdversarialUNetAEBase


class AdversarialUNetAE128(AdversarialUNetAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialUNetAE128
config_path = "configs/adversarial/adversarial_unet_like_ae_128.json"