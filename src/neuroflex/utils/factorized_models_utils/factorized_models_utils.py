import transformers

from exporch import Config

import neuroflex
from neuroflex.factorization.factorized_model import (
    LocalSVDModel,
    GlobalBaseModel,
    GlobalFixedBaseModel,
    GLAMSVDModel,
    LocalHadamardModel,
    update_config_with_model_parameters
)


regularized_training_keys = [
    "initial_regularization_weight",
    "max_regularization_weight",
    "start_step_regularization",
    "steps_regularization_weight_resets"
]

alternative_architectures_keys = [
    "from_pretrained",
    "mapping_layer_name_key",
    "preserve_original_model",
    "remove_average",
    "target_layers",
    "use_names_as_keys",
    "verbose"
]

glam_svd_keys = [
    "pruning_interval",
    "pruning_threshold",
    "thresholding_strategy",
    "pruning_strategy",
    "minimum_number_of_global_layers"
]


def get_factorized_model(
        model: transformers.AutoModel | transformers.PreTrainedModel,
        factorization_method: str,
        config: Config,
) -> neuroflex.GlobalDependentModel:
    """
    Returns the factorized version of the model based on the given factorization method.

    Args:
        model (transformers.AutoModel | transformers.PreTrainedModel):
            The original model to factorize.
        factorization_method (str):
            The factorization method to use.
        config (Config):
            The configuration parameters for the experiment.

    Returns:
        neuroflex.factorization.factorized_model.GlobalDependentModel:
            Factorized version of the model based on the given factorization method.
    """

    alternative_architectures_arguments = config.get_dict(alternative_architectures_keys)
    factorization_method = factorization_method.lower()
    print(alternative_architectures_arguments)
    
    if factorization_method.endswith("model"):
        factorization_method = factorization_method[:-5]

    if factorization_method == "localsvd":
        factorized_model = LocalSVDModel(model, **alternative_architectures_arguments)
    elif factorization_method == "globalbase":
        global_base_keys = ["initialization_type", "average_svd_initialization", "post_init_train"]
        alternative_architectures_arguments.update(config.get_dict(global_base_keys))
        factorized_model = GlobalBaseModel(model, **alternative_architectures_arguments)
    elif factorization_method == "globalfixedbase":
        alternative_architectures_arguments.update(config.get_dict(["initialization_type"]))
        factorized_model = GlobalFixedBaseModel(model, **alternative_architectures_arguments)
    elif factorization_method == "glamsvd":
        regularized_training_arguments = config.get_dict(regularized_training_keys)
        glam_svd_training_arguments = config.get_dict(glam_svd_keys)
        factorized_model = GLAMSVDModel(
            model, **alternative_architectures_arguments, **regularized_training_arguments, **glam_svd_training_arguments)
    elif factorization_method == "hadamard":
        factorized_model = LocalHadamardModel(model, **alternative_architectures_arguments)
    else:
        raise ValueError("Factorization method not recognized")

    try:
        update_config_with_model_parameters(config, factorized_model)
    except Exception as e:
        print(f"Error updating the configuration with the model parameters: {e}")
        pass

    return factorized_model
