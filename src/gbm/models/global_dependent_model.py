from copy import deepcopy

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from gbm.layers.global_dependent_layer import GlobalBaseLinear
from gbm.layers.global_dependent_layer import LocalSVDLinear
from gbm.layers.global_dependent_layer import GlobalFixedRandomBaseLinear


def count_parameters(
        model: nn.Module
) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GlobalDependentModel(ABC, nn.Module):
    """
    Model with global layers used by other layers.
    """

    def __init__(
            self,
            target_model: nn.Module,
            **kwargs
    ) -> None:
        """
        Initialize the model with global layers.

        Args:
            target_model (PreTrainedModel):
                Pretrained model.
            **kwargs:
                Additional keyword arguments.
        """

        super(GlobalDependentModel, self).__init__()

        self.model = deepcopy(target_model)

        self.global_layers = nn.ModuleDict()
        self.conversions = self.define_conversion(**kwargs)
        self._convert_into_global_dependent_model(self.model, **kwargs)

    @abstractmethod
    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Define the conversion of layers into global-dependent layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-dependent layer classes.
        """

    def _convert_into_global_dependent_model(
            self,
            model_tree: nn.Module,
            **kwargs
    ) -> None:
        """
        Converts layers into global-dependent versions.

        Args:
            model_tree (nn.Module):
                Model or module containing layers.
            **kwargs:
                Additional keyword arguments.
        """

        for key in model_tree._modules.keys():
            child = model_tree._modules[key]
            if len(child._modules) == 0:
                if type(child) in self.conversions.keys():
                    model_tree._modules[key] = self.conversions[type(child)](
                        child,
                        self.global_layers,
                        **kwargs)

            else:
                self._convert_into_global_dependent_model(child, **kwargs)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None
    ) -> None:
        """
        Forward pass of the model.

        Args:
            input_ids:
                Input IDs.
            attention_mask:
                Attention mask.

        Returns:
            Output tensor.
        """

        output = self.model.forward(input_ids, attention_mask=attention_mask)

        return output


class GlobalBaseModel(GlobalDependentModel):
    """
    Model with global base layers replacing the linear layers.
    """

    def __init__(
            self,
            pretrained_model,
            rank: int,
            **kwargs
    ) -> None:
        """
        Initialize the global base model.

        Args:
            pretrained_model (PreTrainedModel):
                Pretrained model.
            **kwargs:
                Additional keyword arguments.
        """

        kwargs["rank"] = rank
        super(GlobalBaseModel, self).__init__(pretrained_model, **kwargs)

        self.rank = rank

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Define the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {nn.Linear: GlobalBaseLinear}

        return conversions


class LocalSVDModel(GlobalDependentModel):
    """
    Model with global base layers replacing the linear layers.
    """

    def __init__(
            self,
            pretrained_model,
            rank: int,
            **kwargs
    ) -> None:
        """
        Initialize the global base model.

        Args:
            pretrained_model (PreTrainedModel):
                Pretrained model.
            **kwargs:
                Additional keyword arguments.
        """

        kwargs["rank"] = rank
        super(LocalSVDModel, self).__init__(pretrained_model, **kwargs)

        self.rank = rank

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Define the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {nn.Linear: LocalSVDLinear}

        return conversions


class GlobalFixedRandomBaseModel(GlobalDependentModel):
    """
    Model with global random fixed base layers replacing the linear layers.
    """

    def __init__(
            self,
            pretrained_model,
            **kwargs
    ) -> None:
        """
        Initialize the global base model.

        Args:
            pretrained_model (PreTrainedModel):
                Pretrained model.
            **kwargs:
                Additional keyword arguments.
        """

        super(GlobalFixedRandomBaseModel, self).__init__(pretrained_model, **kwargs)

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Define the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {nn.Linear: GlobalFixedRandomBaseLinear}

        return conversions


import torch
from transformers import BertModel

if __name__ == "__main__":
    # Load the original BERT model
    original_model = BertModel.from_pretrained("bert-base-uncased")

    # Create the global model
    global_model = LocalSVDModel(original_model, rank=768)

    print("Original model:")
    print(original_model)
    print("##################################################")
    print("Global model:")
    print(global_model)
    print("##################################################")
    print("Number of parameters original model:", count_parameters(original_model))
    print("Number of parameters global model:", count_parameters(global_model))
    print("Percentage of parameters:", count_parameters(global_model) / count_parameters(original_model))

    # Input tensor
    input_ids = torch.ones((1, 512)).long()  # Ensure input tensor is of type torch.LongTensor

    # Output of original model
    print("Output of original model:")
    print(original_model(input_ids))
    print("##################################################")

    # Output of global model
    print("Output of global model:")
    print(global_model(input_ids))

