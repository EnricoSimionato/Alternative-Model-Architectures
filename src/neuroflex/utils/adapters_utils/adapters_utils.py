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
    Returns the adapted model to be used in the task.

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

    if config.get("adapter_method").lower() == "lora":
        peft_config = LoraConfig(
            r=config.get("lora_rank"),
            lora_alpha=config.get("lora_alpha"),
            target_modules=config.get("target_modules"),
            lora_dropout=config.get("lora_dropout"),
            bias=config.get("bias"),
            task_type=config.get("task_type")
        )
        model = prepare_model_for_kbit_training(model)
        adapted_model = get_peft_model(
            model,
            peft_config
        )
    elif config.get("adapter_method").lower() == "vera":
        peft_config = VeraConfig(
            r=config.get("lora_rank"),
            target_modules=config.get("target_modules"),
            projection_prng_key=config.get("projection_prng_key"),
            vera_dropout=config.get("lora_dropout"),
            bias=config.get("bias"),
            task_type=config.get("task_type")
        )
        model = prepare_model_for_kbit_training(model)
        adapted_model = get_peft_model(
            model,
            peft_config
        )
    elif config.get("adapter_method").lower().replace("_", "") == "kfcalphalora":
        peft_config = KFCLoraConfig(
            r=config.get("lora_rank"),
            lora_alpha=config.get("lora_alpha"),
            target_modules=config.get("target_modules"),
            lora_dropout=config.get("lora_dropout"),
            bias=config.get("bias"),
            task_type=config.get("task_type")
        )
        model = prepare_model_for_kbit_training(model)
        adapted_model = get_peft_model(
            model,
            peft_config
        )
    else:
        raise ValueError("Invalid adapter method")

    return adapted_model
