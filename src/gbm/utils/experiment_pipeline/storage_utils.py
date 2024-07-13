from __future__ import annotations

import os
import json

from torch import nn

import transformers

import peft
from peft import PeftModel

from gbm.utils.experiment_pipeline.config import Config


def store_model_and_info(
        model: [nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
        config: Config,
        tokenizer: transformers.AutoTokenizer = None,
        store_model_function: callable = None,
        verbose: bool = True
) -> None:
    """
    Stores the model, tokenizer, hyperparameters and stats in the specified path.

    Args:
        model (transformers.AutoModel):
            The model to be stored.
        config (dict):
            The configuration parameters to be stored.
        tokenizer (transformers.AutoTokenizer, optional):
            The tokenizer to be stored. Defaults to None.
        store_model_function (callable, optional):
            The function to store the model. Defaults to None.
        verbose (bool, optional):
            Whether to print the paths where the model, tokenizer, hyperparameters and stats are stored.
            Defaults to True.

    Raises:
        Exception:
            If the specified path does not exist.
    """

    # Check if the directories exist, if not create them
    if tokenizer is not None and not os.path.exists(config.get("path_to_tokenizer")):
        os.makedirs(config.get("path_to_tokenizer"))
    if not os.path.exists(config.get("path_to_model")):
        os.makedirs(config.get("path_to_model"))
    if not os.path.exists(config.get("path_to_configuration")):
        os.makedirs(config.get("path_to_configuration"))

    # Storing the tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(config.get("path_to_tokenizer"))
    # Storing the model
    if store_model_function is not None:
        store_model_function(
            model,
            config.get("path_to_model"),
            config
        )
    else:
        model.save_pretrained(config.get("path_to_model"))

    if verbose:
        if tokenizer is not None:
            print(f"Tokenizer saved in: '{config.get('path_to_tokenizer')}'")
        print(f"Model saved in: '{config.get('path_to_model')}'")

    # Convert config to dictionary if it is not already
    config_dict = config.__dict__

    with open(os.path.join(config.get("path_to_configuration"), 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    if verbose:
        print(f"Experiment configuration parameters saved in: '{os.path.join(config.get('path_to_configuration'), 'config.json')}'")
        print()

    print("Stored model and info")
    print()


def load_model_and_info(
        path_to_experiment: str,
        load_model_function: callable = None,
        verbose: bool = True
) -> tuple:
    """
    Loads the model and info from the specified path.

    Args:
        path_to_experiment (str):
            The path where the model and info are stored, the experiment directory has to contain at least the
            configuration directory containing a dump of the configuration of the experiment.
        load_model_function (callable, optional):
            The function to load the model. Defaults to None.
        verbose (bool, optional):
            Whether to print the paths from which the model, tokenizer and hyperparameters are loaded.
            Defaults to True.

    Returns:
        tuple:
            The model, configuration parameters and tokenizer.
    """

    if not os.path.exists(path_to_experiment):
        raise Exception(f"The specified path does not exist: '{path_to_experiment}'")

    if verbose:
        print(f"Loading model and info from: '{path_to_experiment}'")
        print()

    # Loading configuration parameters
    config = Config(os.path.join(path_to_experiment, "configuration", "config.json"))
    if verbose:
        print(f"Configuration parameters loaded from: '{os.path.join(path_to_experiment, 'configuration')}'")

    # Loading the model
    if load_model_function is not None:
        model = load_model_function(
            config.get("path_to_model"),
            config
        )
    else:
        if config.get("task") == "chatbot":
            model = transformers.AutoModelForCausalLM.from_pretrained(config.get("path_to_model"))
        elif config.get("task") == "classification":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(config.get("path_to_model"))
        else:
            model = transformers.AutoModel.from_pretrained(config.get("path_to_model"))
    if verbose:
        print(f"Model loaded from: '{config.get('path_to_model')}'")

    # Loading the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.get("path_to_tokenizer"))
    if verbose:
        print(f"Tokenizer loaded from: '{config.get('path_to_tokenizer')}'")

    print("Loaded model and info")
    print()

    return model, config, tokenizer


def load_peft_model_function(
        path_to_adapters: str,
        config: Config
) -> peft.PeftModel:
    """
    Loads the PEFT model from the specified path.

    Args:
        path_to_adapters (str):
            The path to the adapters.
        config (Config):
            The configuration object.

    Returns:
        peft.PeftModel:
            The PEFT model.
    """

    original_model = config.get_original_model()

    return PeftModel.from_pretrained(
        original_model,
        path_to_adapters
    )
