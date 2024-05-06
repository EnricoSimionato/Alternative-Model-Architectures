import torch
from torch import nn


def count_parameters(
        model: nn.Module,
        only_trainable: bool = False
) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module):
            The model whose parameters are to be counted.
        only_trainable (bool, optional):
            Whether to count only the trainable parameters. Defaults to False.

    Returns:
        int:
            The number of parameters in the model.
    """

    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def set_parameters_count_in_config(
        config: dict,
        original_model: torch.nn.Module,
        gd_model: torch.nn.Module
) -> dict:
    """
    Sets the number of parameters of the model and related information in the configuration dictionary.

    Args:
        config (dict):
            The dictionary containing the configuration parameters.
        original_model (torch.nn.Module):
            The original model.
        gd_model (torch.nn.Module):
            The model with global dependent layers.

    Returns:
        dict:
            The configuration dictionary with the number of parameters and related information.
    """

    original_model_parameters = count_parameters(original_model)
    model_parameters = count_parameters(gd_model)

    config["original_model_parameters"] = original_model_parameters
    config["model_trainable_parameters"] = model_parameters
    config["percentage_parameters"] = model_parameters / original_model_parameters * 100
    config["model_parameters"] = count_parameters(gd_model, only_trainable=True)

    return config
