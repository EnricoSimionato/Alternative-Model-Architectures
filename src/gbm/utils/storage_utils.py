import os
from datetime import datetime

import pickle

import transformers


def store_model_and_info(
        model_name: str,
        model: transformers.AutoModel,
        tokenizer: transformers.AutoTokenizer,
        hyperparameters: dict,
        losses: dict,
        path: str = None,
        verbose: bool = True
):
    """
    Stores the model, tokenizer, hyperparameters and stats in the specified path.

    Args:
        model_name (str):
            The base name to use to store the model and its information.
        model (transformers.AutoModel):
            The model to be stored.
        tokenizer (transformers.AutoTokenizer):
            The tokenizer to be stored.
        hyperparameters (dict):
            The hyperparameters to be stored.
        losses (dict):
            The losses to be stored.
        path (str, optional):
            The path where the model, tokenizer, hyperparameters and stats are to be stored.
            Defaults to None.
        verbose (bool, optional):
            Whether to print the paths where the model, tokenizer, hyperparameters and stats are stored.
            Defaults to True.

    Returns:
        str:
            The name of the stored model.

    Raises:
        Exception:
            If the specified path does not exist.
    """

    if path is None:
        path = os.getcwd()

    if not os.path.exists(path):
        raise Exception(f"Path '{path}' does not exist.")

    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tokenizer_name = f"{model_name}_tokenizer_{date}"
    model_name = f"{model_name}_model_{date}"

    if not os.path.exists(os.path.join(path, "models")):
        os.makedirs(os.path.join(path, "models"))
    path_to_tokenizer = os.path.join(path, "models", tokenizer_name)
    path_to_model = os.path.join(path, "models", model_name)

    # Storing the tokenizer
    tokenizer.save_pretrained(path_to_tokenizer)
    # Storing the model
    model.save_pretrained(path_to_model)

    if verbose:
        print(f"Tokenizer saved in: '{path_to_tokenizer}'")
        print(f"Model saved in: '{path_to_model}'")

    if not os.path.exists(os.path.join(path, "stats")):
        os.makedirs(os.path.join(path, "stats"))
    path_to_stats = os.path.join(path, "stats", model_name)
    with open(path_to_stats, 'wb') as f:
        pickle.dump(losses, f)

    if verbose:
        print(f"Stats saved in: '{path_to_stats}'")

    if not os.path.exists(os.path.join(path, "hyperparameters")):
        os.makedirs(os.path.join(path, "hyperparameters"))
    path_to_hyperparameters = os.path.join(path, "hyperparameters", model_name)
    with open(path_to_hyperparameters, 'wb') as f:
        pickle.dump(hyperparameters, f)

    if verbose:
        print(f"Hyperparameters saved in: '{path_to_hyperparameters}'")

    return model_name


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
        dict:
            The losses of the model when trained.
    """

    if path is None:
        path = os.getcwd()

    if not os.path.exists(path):
        raise Exception(f"Path '{path}' does not exist.")

    if not os.path.exists(os.path.join(path, "models")):
        raise Exception(f"Path '{os.path.join(path, 'models')}' does not exist.")

    if not os.path.exists(os.path.join(path, "stats")):
        raise Exception(f"Path '{os.path.join(path, 'stats')}' does not exist.")

    if not os.path.exists(os.path.join(path, "hyperparameters")):
        raise Exception(f"Path '{os.path.join(path, 'hyperparameters')}' does not exist.")

    tokenizer_name = model_name.replace("model", "tokenizer")

    path_to_tokenizer = os.path.join(path, "models", tokenizer_name)
    path_to_model = os.path.join(path, "models", model_name)

    # Loading the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(path_to_tokenizer)
    # Loading the model
    model = transformers.AutoModel.from_pretrained(path_to_model)

    if verbose:
        print(f"Tokenizer loaded from: '{path_to_tokenizer}'")
        print(f"Model loaded from: '{path_to_model}'")

    path_to_stats = os.path.join(path, "stats", model_name)
    with open(path_to_stats, 'rb') as f:
        losses = pickle.load(f)

    if verbose:
        print(f"Stats loaded from: '{path_to_stats}'")

    path_to_hyperparameters = os.path.join(path, "hyperparameters", model_name)
    with open(path_to_hyperparameters, 'rb') as f:
        hyperparameters = pickle.load(f)

    if verbose:
        print(f"Hyperparameters loaded from: '{path_to_hyperparameters}'")
        print()

    if print_info:
        print("SETTING OF THE TRAINING - HYPERPARAMETERS:")
        for key, value in hyperparameters.items():
            print(f"{key}: {value}")
        print()

    return model, tokenizer, hyperparameters, losses


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
