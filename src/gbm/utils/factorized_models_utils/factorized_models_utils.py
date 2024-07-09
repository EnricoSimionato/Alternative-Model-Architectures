import transformers

import gbm
from gbm import (
    GlobalBaseModel,
    LocalSVDModel
)

from gbm.utils.pipeline.config import Config


def get_factorized_model(
        model: transformers.AutoModel,
        config: Config,
) -> gbm.GlobalDependentModel:
    """
    Returns the factorized model to use in the experiment.

    Args:
        config (Config):
            The configuration parameters for the experiment.

    Returns:
        gbm.GlobalDependentModel:
            The factorized model to use in the experiment.
    """

    if not config.contains("factorization_method"):
        raise ValueError("Factorization method not specified")

    keys = [
        "target_layers",
        "use_names_as_keys",
        "mapping_layer_name_key",
        "remove_average",
        "from_pretrained",
        "preserve_original_model",
        "verbose"
    ]
    arguments = {}
    for key in keys:
        if config.contains(key):
            arguments[key] = config.get(key)

    if config.get("factorization_method").lower() == "globalbase":
        return GlobalBaseModel(
            model,
            **arguments
        )
    elif config.get("factorization_method").lower() == "localsvd":
        return LocalSVDModel(
            model,
            **arguments
        )
    else:
        raise ValueError("Factorization method not recognized")
