import torch
from transformers import AutoModelForCausalLM

from gbm import StructureSpecificGlobalDependentLinear, StructureSpecificGlobalDependent
from gbm.utils import check_model_for_nan
from src.gbm.models.global_dependent_model import GLAMSVDModel, PruningStrategy

if __name__ == "__main__":
    # Load the original BERT model
    model_id = "bert-base-uncased"
    original_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    print(original_model)

    # Create the global model
    global_model = GLAMSVDModel(
        original_model,
        target_layers={
            #"word_embeddings": {"rank": 128},
            "query": {"rank": 64},
            #"key": {"rank": 64},
            #"value": {"rank": 64},
            #"dense": {"rank": 64},
        },
        use_names_as_keys=True,
        preserve_original_model=True,
        verbose=True,
        initial_regularization_weight=0.01,
        max_regularization_weight=1.0,
        start_step_regularization=0,
        steps_regularization_weight_resets=100,
        pruning_interval=100,
        pruning_threshold=2.0,
        pruning_strategy=PruningStrategy.AVERAGE
    )

    #check_model_for_nan(global_model)

    #print(global_model)

    global_model.prune_global_layers()

    print(global_model)

    print(global_model.model.bert.encoder.layer[4].attention.self.query.structure)
    print(global_model.model.bert.encoder.layer[8].attention.self.query.structure)
