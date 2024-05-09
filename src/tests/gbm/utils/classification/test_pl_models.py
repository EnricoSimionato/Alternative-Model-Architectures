from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from gbm.utils.classification.pl_datasets import IMDBDataModule
from gbm.utils.classification.pl_models import ClassifierModelWrapper

CONFIG = {
    "model": "bert-base-uncased",
    "tokenizer": "bert-base-uncased",
    "max_length": 32,
    "batch_size": 1,
    "num_workers": 1,
    "split": (0.8, 0.1, 0.1),
    "num_labels": 2,
    "learning_rate": 2e-5,
    "epochs": 3,
    "device": "mps",
    "seed": 42
}

class TestClassifierModelWrapper:

    def test_init(self):
        model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer"])
        self.model = ClassifierModelWrapper(
            model=model,
            tokenizer=self.tokenizer,

        )
        self.data = IMDBDataModule(
            "",
            CONFIG["batch_size"],
            CONFIG["num_workers"],
            self.tokenizer,
            CONFIG["max_length"],
            CONFIG["split"],
            CONFIG["seed"]
        )
    def test_fit(self):
        pass

    def test_predict(self):
        pass

    def test_predict_proba(self):
        pass

    def test_get_params(self):
        pass

    def test_set_params


if __name__ = "__main__":
    pass