import transformers

import gbm
from gbm import (
    LocalSVDModel,
    GlobalBaseModel,
    GlobalFixedBaseModel,
    GLAMSVDModel
)

from gbm.utils.experiment_pipeline.config import Config
from gbm.models.global_dependent_model import update_config_with_model_parameters

regularized_training_keys = [
    "initial_regularization_weight",
    "max_regularization_weight",
    "start_step_regularization",
    "steps_regularization_weight_resets"
]

alternative_architectures_keys = [
    "target_layers",
    "use_names_as_keys",
    "mapping_layer_name_key",
    "remove_average",
    "from_pretrained",
    "preserve_original_model",
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
        model: transformers.AutoModel,
        config: Config,
) -> gbm.GlobalDependentModel:
    """
    Returns the factorized model to use in the experiment.

    Args:
        model (transformers.AutoModel):
            The original model to factorize.
        config (Config):
            The configuration parameters for the experiment.

    Returns:
        gbm.GlobalDependentModel:
            The factorized model to use in the experiment.
    """

    if not config.contains("factorization_method"):
        raise ValueError("Factorization method not specified")

    alternative_architectures_arguments = config.get_dict(alternative_architectures_keys)
    factorization_method = config.get("factorization_method").lower()

    if factorization_method.endswith("model"):
        factorization_method = factorization_method[:-5]

    if factorization_method == "localsvd":
        factorized_model = LocalSVDModel(
            model,
            **alternative_architectures_arguments
        )
    elif factorization_method == "globalbase":
        alternative_architectures_arguments.update(config.get_dict(["initialization_type"]))
        factorized_model = GlobalBaseModel(
            model,
            **alternative_architectures_arguments
        )
    elif factorization_method == "globalfixedbase":
        alternative_architectures_arguments.update(config.get_dict(["initialization_type"]))
        factorized_model = GlobalFixedBaseModel(
            model,
            **alternative_architectures_arguments
        )
    elif factorization_method == "glamsvd":
        regularized_training_arguments = config.get_dict(regularized_training_keys)
        glam_svd_training_arguments = config.get_dict(glam_svd_keys)
        factorized_model = GLAMSVDModel(
            model,
            **alternative_architectures_arguments,
            **regularized_training_arguments,
            **glam_svd_training_arguments
        )
    else:
        raise ValueError("Factorization method not recognized")

    update_config_with_model_parameters(
        config,
        factorized_model
    )

    return factorized_model
