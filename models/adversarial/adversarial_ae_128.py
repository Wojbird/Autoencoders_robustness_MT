from models.adversarial.adversarial_ae_base import AdversarialAEBase


class AdversarialAE128(AdversarialAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialAE128
config_path = "configs/adversarial/adversarial_ae_128.json"