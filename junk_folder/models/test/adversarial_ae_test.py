from models.adversarial.adversarial_ae_base import AdversarialAEBase


class AdversarialAETest(AdversarialAEBase):
    def __init__(self, config: dict):
        super().__init__(config)


model_class = AdversarialAETest
config_path = "configs/test/adversarial_ae_test.json"