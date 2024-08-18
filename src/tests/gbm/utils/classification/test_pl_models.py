import pytorch_lightning as pl

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from neuroflex.utils.classification.pl_datasets import IMDBDataModule
from neuroflex.utils.classification.pl_models import ClassifierModelWrapper
from neuroflex.utils.device_utils import get_available_device

CONFIG = {
    "model": "bert-base-uncased",
    "tokenizer": "bert-base-uncased",
    "max_length": 32,
    "batch_size": 4,
    "num_workers": 1,
    "split": (0.8, 0.1, 0.1),
    "num_classes": 2,
    "id2label": {0: "negative", 1: "positive"},
    "label2id": {"negative": 0, "positive": 1},
    "learning_rate": 2e-5,
    "gradient_accumulation_steps": 1,
    "max_epochs": 3,
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
            num_classes=CONFIG["num_classes"],
            id2label=CONFIG["id2label"],
            label2id=CONFIG["label2id"],
            learning_rate=CONFIG["learning_rate"],
            max_epochs=CONFIG["num_epochs"],
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


if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model"])
    model.to(CONFIG["device"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer"])

    dataset = IMDBDataModule(
        "",
        CONFIG["batch_size"],
        CONFIG["num_workers"],
        tokenizer,
        CONFIG["max_length"],
        CONFIG["split"],
        CONFIG["seed"]
    )

    # Instantiating ClassifierModelWrapper object in order to train the model
    lightning_model = ClassifierModelWrapper(
        model=model,
        tokenizer=tokenizer,

        num_classes=CONFIG["num_classes"],
        id2label=CONFIG["id2label"],
        label2id=CONFIG["label2id"],

        learning_rate=CONFIG["learning_rate"],
        max_epochs=CONFIG["max_epochs"],
    )

    # Defining early stopping callback
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="validation_loss",
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min",
    )

    # Defining trainer settings
    lightning_trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        val_check_interval=0.5,
        accumulate_grad_batches=CONFIG["gradient_accumulation_steps"],
        callbacks=[
            early_stopping_callback,
        ],
        accelerator=get_available_device().type,
        deterministic=True,
    )

    # Computing the validation error of the model before training starts
    #lightning_trainer.validate(lightning_model, dataset)

    lightning_trainer.fit(lightning_model, dataset)
