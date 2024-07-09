import transformers

from peft import (
    LoraConfig,
    VeraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from gbm.utils.pipeline.config import Config


def get_adapted_model(
        model: transformers.AutoModel,
        config: Config
) -> transformers.AutoModel:
    """
    Returns the adapted model to be used in the task.

    Args:
        model (transformers.AutoModel):
            The original model to be adapted.
        config (dict):
            The configuration parameters to use in the adaptation.

    Returns:
        peft.PeftModel
            The adapted model.
    """

    if config.contains("adapter_method"):
        if config.get("adapter_method").lower() == "lora":
            lora_config = LoraConfig(
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
                lora_config
            )

            return adapted_model
        elif config.get("adapter_method").lower() == "vera":
            lora_config = VeraConfig(
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
                lora_config
            )

            return adapted_model
        else:
            raise ValueError("Invalid adapter method")