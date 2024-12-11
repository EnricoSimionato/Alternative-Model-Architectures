from abc import ABC, abstractmethod

import torch

import transformers


class FactorizedModel(ABC):
    """
    Factorized model.
    """

    @abstractmethod
    def get_model(
            self
    ) -> None | torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
        """
        Returns the model.

        Returns:
            None | torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                Model.
        """

        return None