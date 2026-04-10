from models.adversarial.adversarial_ae_base import AdversarialAEBase


class AdversarialAE512(AdversarialAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialAE512
config_path = "configs/adversarial/adversarial_ae_512.json"