from models.adversarial.adversarial_ae_base import AdversarialAEBase


class AdversarialAE64(AdversarialAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialAE64
config_path = "configs/adversarial/adversarial_ae_64.json"