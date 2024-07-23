from __future__ import annotations

import torch
import transformers


def check_model_for_nan(
        model: [transformers.PreTrainedModel | transformers.AutoModel]
) -> bool:
    """
    Check if there are NaNs in the model parameters or gradients.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model to check for NaNs.

    Returns:
        bool:
            True if there are NaNs in the model parameters or gradients, False otherwise.
    """

    nan_in_params = False
    nan_in_grads = False

    for name, param in model.named_parameters():

        if torch.isnan(param).any():
            print(f'NaN found in parameter: {name}')
            nan_in_params = True
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f'NaN found in gradient: {name}')
            nan_in_grads = True

    if not nan_in_params:
        print('No NaNs found in model parameters.')
    if not nan_in_grads:
        print('No NaNs found in model gradients.')

    return nan_in_params or nan_in_grads
