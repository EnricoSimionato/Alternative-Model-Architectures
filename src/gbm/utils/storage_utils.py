import os
import pickle

import transformers

import peft

from gbm.utils.pipeline.config import Config


def store_model_and_info(
        model: transformers.AutoModel,
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
            The tokenizer to be stored.
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
            config.get("path_to_model")
        )
    else:
        model.save_pretrained(config.get("path_to_model"))

    if verbose:
        if tokenizer is not None:
            print(f"Tokenizer saved in: '{config.get('path_to_tokenizer')}'")
        print(f"Model saved in: '{config.get('path_to_model')}'")

    with open(os.path.join(config.get("path_to_configuration"), "config"), 'wb') as f:
        pickle.dump(config, f)

    if verbose:
        print(f"Experiment configuration parameters saved in: '{os.path.join(config.get('path_to_configuration'), 'config')}'")

    print("Stored model and info")


def load_model_and_info():
    pass


def test_store() -> None:
    model = transformers.AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    hyperparameters = {
        "batch_size": 16,
        "learning_rate": 1e-5
    }

    losses = {
        "train": [0.5, 0.4, 0.3],
        "validation": [0.6, 0.5, 0.4],
        "test": [0.7, 0.6, 0.5]
    }

    _ = store_model_and_info("bert", model, tokenizer, hyperparameters, losses, verbose=False)


def test_load() -> None:
    model = transformers.AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    hyperparameters = {
        "batch_size": 16,
        "learning_rate": 1e-5
    }

    losses = {
        "train": [0.5, 0.4, 0.3],
        "validation": [0.6, 0.5, 0.4],
        "test": [0.7, 0.6, 0.5]
    }

    model_name = store_model_and_info("bert", model, tokenizer, hyperparameters, losses, verbose=False)

    load_model_and_info(model_name, print_info=True, verbose=True)


if __name__ == "__main__":
    test_load()
