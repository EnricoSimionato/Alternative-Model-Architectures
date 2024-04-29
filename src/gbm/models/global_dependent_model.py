from copy import deepcopy

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import device

from gbm.layers.global_dependent_layer import (
    MergeableLayer,
    GlobalBaseLinear,
    LocalSVDLinear,
    GlobalFixedBaseLinear,
)

from transformers import BertModel


class GlobalDependentModel(ABC, nn.Module):
    """
    Model with global layers replacing some layers of the model.

    Args:
        target_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
            Layers to factorize.
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        target_layers (dict):
            Layers to factorize.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers.
        model (PreTrainedModel):
            Pretrained model.
        global_layers (nn.ModuleDict):
            Global layers.
        conversions (dict):
            Mapping of layer types to global-dependent layer classes.
    """

    def __init__(
            self,
            target_model: nn.Module,
            target_layers: dict,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            **kwargs
    ) -> None:
        super(GlobalDependentModel, self).__init__()

        self.target_layers = target_layers
        if mapping_layer_name_key is None and use_names_as_keys:
            self.mapping_layer_name_key = {layer_name: layer_name for layer_name in target_layers.keys()}
        else:
            self.mapping_layer_name_key = mapping_layer_name_key
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
        Defines the conversion of layers into global-dependent layers.

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

        for layer_name in model_tree._modules.keys():
            child = model_tree._modules[layer_name]
            if len(child._modules) == 0:
                if (type(child) in self.conversions.keys() and
                        layer_name in self.target_layers.keys()):

                    kwargs_layer = kwargs.copy()
                    kwargs_layer.update(self.target_layers[layer_name])

                    model_tree._modules[layer_name] = self.conversions[type(child)](
                        child,
                        self.global_layers,
                        target_name=None if self.mapping_layer_name_key is None else self.mapping_layer_name_key[layer_name],
                        **kwargs_layer
                    )

            else:
                self._convert_into_global_dependent_model(child, **kwargs)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            **kwargs
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

        output = self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        return output

    def merge(
            self,
            layers_to_merge: tuple = None,
            **kwargs
    ) -> nn.Module:
        """
        Merges the global layers into the model and returns the result.

        Args:
            layers_to_merge (list):
                List of names of layers to merge.
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Model with global layers merged.
        """

        if layers_to_merge is None:
            layers_to_merge = tuple(self.target_layers.keys())

        merged_model = deepcopy(self.model)
        self._merge_model(merged_model, layers_to_merge, **kwargs)

        return merged_model

    def _merge_model(
            self,
            model_tree: nn.Module,
            layers_to_merge: tuple,
            **kwargs
    ) -> None:
        """
        Utility function to merge the global layers into the model.

        Args:
            model_tree (nn.Module):
                Module containing layers to merge.
            layers_to_merge (list):
                List of names of layers to merge.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in model_tree._modules.keys():
            child = model_tree._modules[layer_name]
            if (isinstance(child, MergeableLayer)) and layer_name in layers_to_merge:
                model_tree._modules[layer_name] = child.merge(**kwargs)
            elif len(child._modules) == 0:
                pass
            else:
                self._merge_model(child, layers_to_merge, **kwargs)

    @property
    def device(self) -> device:
        """
        Device where the model is located.

        Returns:
            Device.
        """

        return next(self.parameters()).device


class LocalSVDModel(GlobalDependentModel):
    """
    Model with LocalSVDLinear layers replacing linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model,
            target_layers: dict,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            **kwargs
    ) -> None:
        super(LocalSVDModel, self).__init__(
            pretrained_model,
            target_layers,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            **kwargs
        )

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {nn.Linear: LocalSVDLinear}

        return conversions


class GlobalBaseModel(GlobalDependentModel):
    """
    Model with GlobalBaseLinear layers replacing linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model,
            target_layers: dict,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            **kwargs
    ) -> None:
        super(GlobalBaseModel, self).__init__(
            pretrained_model,
            target_layers,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            **kwargs
        )

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


class GlobalFixedBaseModel(GlobalDependentModel):
    """
    Model with GlobalFixedBaseLinear layers replacing the linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model,
            target_layers: dict,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            **kwargs
    ) -> None:
        super(GlobalFixedBaseModel, self).__init__(
            pretrained_model,
            target_layers,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            **kwargs
        )

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {nn.Linear: GlobalFixedBaseLinear}

        return conversions


if __name__ == "__main__":
    # Load the original BERT model
    original_model = BertModel.from_pretrained("bert-base-uncased")

    # Create the global model
    global_model = GlobalBaseModel(
        original_model,
        target_layers={
            "query": {"rank": 78},
            "value": {"rank": 78},
        },
        use_names_as_keys=False,
        #mapping_layer_name_key={
        #    "query": "attention",
        #    "value": "attention",
        #},
        rank=78,
    )

    """
    print("Original model:")
    print(original_model)
    print("##################################################")
    """

    print("Global model:")
    print(global_model)
    """
    print("##################################################")
    print("Number of parameters original model:", count_parameters(original_model))
    print("Number of parameters global model:", count_parameters(global_model))
    print("Percentage of parameters:", count_parameters(global_model) / count_parameters(original_model))
    """
    #print("Device of the model:", global_model.device)

    """
    # Input tensor
    input_ids = torch.ones((1, 512)).long()  # Ensure input tensor is of type torch.LongTensor

    # Output of original model
    print("Output of original model:")
    print(original_model(input_ids))
    print("##################################################")

    # Output of global model
    print("Output of global model:")
    print(global_model.forward(input_ids))
    print"
    """
    print()
