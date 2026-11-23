from models.adversarial_unet.adversarial_ae_base import AdversarialAEBase


class AdversarialAE256(AdversarialAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialAE256
config_path = "configs/adversarial_unet/adversarial_ae_256.json"