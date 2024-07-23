from __future__ import annotations

import transformers


def count_parameters(
        model: [transformers.PreTrainedModel | transformers.AutoModel],
        only_trainable: bool = False
) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
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
