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
