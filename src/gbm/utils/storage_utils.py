import os

import pickle

import transformers

from gbm.utils.pipeline.config import Config


def store_model_and_info(
        model: transformers.AutoModel,
        config: Config,
        tokenizer: transformers.AutoTokenizer = None,
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


def load_model_and_info(
        model_name: str,
        path: str = None,
        print_info: bool = True,
        verbose: bool = True
):
    """
    Loads the model, tokenizer, hyperparameters and stats from the specified path.

    Args:
        model_name (str):
            The base name to use to load the model and its information.
        path (str, optional):
            The path where the model, tokenizer, hyperparameters and stats are stored.
            Defaults to None.
        print_info (bool, optional):
            Whether to print the hyperparameters and stats.
            Defaults to True.
        verbose (bool, optional):
            Whether to print the paths where the model, tokenizer, hyperparameters and stats are loaded from.
            Defaults to True.

    Returns:
        transformers.AutoModel:
            The loaded model.
        transformers.AutoTokenizer:
            The loaded tokenizer.
        dict:
            The hyperparameters used in the training of the model.
    """

    if path is None:
        path = os.getcwd()

    if not os.path.exists(path):
        raise Exception(f"Path '{path}' does not exist.")

    # Check if the directories exist
    if not os.path.exists(os.path.join(path, "models")):
        raise Exception(f"Path '{os.path.join(path, 'models')}' does not exist.")
    if not os.path.exists(os.path.join(path, "tokenizers")):
        raise Exception(f"Path '{os.path.join(path, 'tokenizers')}' does not exist.")
    if not os.path.exists(os.path.join(path, "stats")):
        raise Exception(f"Path '{os.path.join(path, 'stats')}' does not exist.")
    if not os.path.exists(os.path.join(path, "hyperparameters")):
        raise Exception(f"Path '{os.path.join(path, 'hyperparameters')}' does not exist.")

    # Defining the paths
    path_to_tokenizer = os.path.join(path, "tokenizers", model_name)
    path_to_model = os.path.join(path, "models", model_name)
    path_to_configuration = os.path.join(path, "hyperparameters", model_name)

    # Loading the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(path_to_tokenizer)
    # Loading the model
    model = transformers.AutoModel.from_pretrained(path_to_model)

    if verbose:
        print(f"Tokenizer loaded from: '{path_to_tokenizer}'")
        print(f"Model loaded from: '{path_to_model}'")

    with open(path_to_configuration, 'rb') as f:
        config = pickle.load(f)

    if verbose:
        print(f"Hyperparameters loaded from: '{path_to_configuration}'")
        print()

    if print_info:
        print("SETTING OF THE TRAINING - HYPERPARAMETERS:")
        for key, value in config.items():
            print(f"{key}: {value}")
        print()

    return model, tokenizer, config


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
