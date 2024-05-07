import os


import torch

from transformers import (
    AutoModelForSequenceClassification,
)

from gbm.models.global_dependent_model import GlobalBaseModel

import pytest


class TestGlobalBaseModel:

    def __init__(self):
        self.original_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self.prefix = "GlobalBaseModel"
        self.test_init()

    def test_init(self):
        global_model = GlobalBaseModel(
            self.original_model,
            target_layers={
                "query": {"rank": 78},
                "value": {"rank": 78},
            },
            use_names_as_keys=True
        )
        self.global_model = global_model

    @pytest.mark
    def test_save_pretrained(self):
        self.global_model.save_pretrained(os.path.join(os.getcwd(), f"test_save_pretrained_{self.prefix}"))

    @pytest.mark
    def test_from_pretrained(self):
        self.global_model.save_pretrained(f"test_save_pretrained_{self.prefix}")
        loaded_global_model = GlobalBaseModel.from_pretrained(f"test_save_pretrained_{self.prefix}")
        input_ids = torch.ones((1, 512)).long()
        y = self.global_model(input_ids)
        y_loaded = loaded_global_model(input_ids)

        assert torch.equal(y.logits, y_loaded.logits)
        assert loaded_global_model.target_layers == self.global_model.target_layers
