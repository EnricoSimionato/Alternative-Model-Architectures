from typing import Union

import transformers

from exporch import Config

from peft import PeftModel, PeftMixedModel
from imports.peft import (
    LoraConfig,
    VeraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from imports.peft.tuners.kfc.config import KFCLoraConfig


def get_adapted_model(
        model: transformers.PreTrainedModel,
        config: Config
) -> Union[PeftModel, PeftMixedModel]:
    """
    Wraps and returns the model with the specified adapter method.

    Args:
        model (transformers.PreTrainedModel):
            The original model to be adapted.
        config (dict):
            The configuration parameters to use in the adaptation.

    Returns:
        Union[PeftModel, PeftMixedModel]:
            The adapted model.
    """

    if not config.contains("adapter_method"):
        raise ValueError("Adapter method not specified")

    target_layers = config.get("target_layers")
    if config.get("adapter_method").lower() == "lora":
        model = prepare_model_for_kbit_training(model)
        for target_layer, params in target_layers.items():
            lora_rank = params["rank"]
            lora_alpha = 2*params["rank"]
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=[target_layer,],
                lora_dropout=config.get("lora_dropout"),
                bias=config.get("bias"),
                task_type=config.get("task_type")
            )

            adapted_model = get_peft_model(model, peft_config)

    elif config.get("adapter_method").lower() == "vera":
        model = prepare_model_for_kbit_training(model)
        for target_layer, params in target_layers.items():
            lora_rank = params["rank"]
            peft_config = VeraConfig(
                r=lora_rank,
                target_modules=[target_layer,],
                projection_prng_key=config.get("projection_prng_key"),
                vera_dropout=config.get("lora_dropout"),
                bias=config.get("bias"),
                task_type=config.get("task_type")
            )

        adapted_model = get_peft_model(model, peft_config)

    elif config.get("adapter_method").lower().replace("_", "") == "abaco-lora":
        model = prepare_model_for_kbit_training(model)
        for target_layer, params in target_layers.items():
            lora_rank = params["rank"]
            lora_alpha = 2*params["rank"]
            peft_config = KFCLoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=[target_layer,],
                lora_dropout=config.get("lora_dropout"),
                bias=config.get("bias"),
                task_type=config.get("task_type")
            )

            adapted_model = get_peft_model(model, peft_config)

    else:
        raise ValueError("Invalid adapter method")

    return adapted_model
