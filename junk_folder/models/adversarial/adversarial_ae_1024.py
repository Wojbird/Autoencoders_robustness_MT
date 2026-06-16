from models.adversarial.adversarial_ae_base import AdversarialAEBase


class AdversarialAE1024(AdversarialAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialAE1024
config_path = "configs/adversarial/adversarial_ae_1024.json"