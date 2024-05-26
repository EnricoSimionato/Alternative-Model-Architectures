import os


import torch

from transformers import (
    AutoModelForSequenceClassification, AutoModelForCausalLM,
)

from gbm.models.global_dependent_model import GlobalBaseModel

import pytest

from gbm.utils import check_model_for_nan, count_parameters

"""
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
"""

if __name__ == "__main__":
    # Load the original BERT model
    # model_id = "google/gemma-2b"
    model_id = "bert-base-uncased"
    original_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    print(original_model)

    # Create the global model
    global_model = GlobalBaseModel(
        original_model,
        target_layers={
            "word_embeddings": {"rank": 128},
            "query": {"rank": 64},
            "key": {"rank": 64},
            "value": {"rank": 64},
            "dense": {"rank": 64},
        },
        use_names_as_keys=True,
        remove_average=True,
        preserve_original_model=True,
        verbose=True
    )

    check_model_for_nan(global_model)

    """
    # Create the global model
    global_model = GlobalBaseModel(
        original_model,
        target_layers={
            "word_embeddings": {"rank": 128},
            "query": {"rank": 64},
            "key": {"rank": 64},
            "value": {"rank": 64},
            "dense": {"rank": 64},
        },
        use_names_as_keys=True,
        #mapping_layer_name_key={
        #    "query": "attention",
        #    "value": "attention",
        #},
        rank=78,
        preserve_original_model=True,
        verbose=True
    )
    """

    # global_model.save_pretrained("global_model")

    print("Original model:")
    print(original_model)
    print("##################################################")

    print(original_model)
    print("Global model:")
    print(global_model)

    print("##################################################")
    print("Number of parameters original model:", count_parameters(original_model))
    print("Number of parameters global model:", count_parameters(global_model))
    print("Percentage of parameters:", count_parameters(global_model) / count_parameters(original_model))
    # print("Device of the model:", global_model.device)

    """
    # Input tensor
    input_ids = torch.ones((1, 512)).long()  # Ensure input tensor is of type torch.LongTensor

    # Output of original model
    print("Output of original model:")
    print(original_model(input_ids))
    print("##################################################")

    # Output of global model
    print("Output of global model:")
    print(global_model.forward(input_ids))
    """

    print()
