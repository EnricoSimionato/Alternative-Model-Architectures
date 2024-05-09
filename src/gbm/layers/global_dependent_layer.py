from typing import Any

from abc import ABC, abstractmethod

import math
import numpy as np

import torch
import torch.nn as nn

from gbm.utils.device_utils import get_available_device


class MergeableLayer:
    """
    Interface class that defines the interface for a mergeable layer.
    A mergeable layer is a layer that can be merged, it has the merge method that returns an equivalent linear layer.
    """

    @abstractmethod
    def merge(
            self,
            **kwargs
    ) -> nn.Module:
        """
        Merges the global and local layers into an equivalent layer.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Equivalent linear layer with merged weights and bias.
        """


class GlobalDependent(ABC, MergeableLayer, nn.Module):
    """
    Abstract class that implements a layer with dependencies on global matrices.
    It implements a linear layer with dependencies on global matrices or combinations of linear layers, for other types
    of layers it has to be extended.

    It has to substitute Linear layers so the interface has to be the same (except for the constructor).

    Args:
        in_features (int):
            Number of input features.
        out_features (int):
            Number of output features.
        global_layers (nn.ModuleDict):
            List of global matrices used in the linear layer.
        structure (dict):
            Structure of the layer. Each element of the list has to contain a tuple with the type of layer ('global' for
            global or 'local' for local) and the key for the global matrix or the configuration for the local matrix.
        bias (bool):
            Flag to include bias in the linear layer.
        *args:
            Variable length argument list for defining input and output features.
        **kwargs:
            Arbitrary keyword arguments for layer configuration.

    Attributes:
        in_features (int):
            Number of input features.
        out_features (int):
            Number of output features.
        global_layers (dict):
            Dictionary containing the global layers.
        local_layers (nn.ModuleDict):
            Dictionary containing the local layers.
        structure (dict):
            Structure of the layer. Each element of the list has to contain a tuple with the type of layer ('global' for
            global or 'local' for local) and the key for the global matrix or the configuration for the local matrix.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            global_layers: nn.ModuleDict,
            structure: dict,
            bias: bool = True,
            *args,
            **kwargs
    ) -> None:

        super(GlobalDependent, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.global_layers = global_layers
        self.local_layers = nn.ModuleDict()

        # Initial sanity check on the structure of the layer
        if len(structure) < 1:
            raise Exception("The structure has to contain at least one layer")
        for layer in structure:
            if layer["scope"] != "global" and layer["scope"] != "local":
                raise Exception("The structure of each layer has to contain 'global' or 'local' as 'scope'")
        self.structure = structure

        self._create_layer(bias, **kwargs)

    def _create_layer(
            self,
            bias: bool,
            **kwargs
    ) -> None:
        """
        Creates the actual layer of the class.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        self._initialize_global_layers(bias, **kwargs)
        self._initialize_local_layers(bias, **kwargs)

    @abstractmethod
    def _initialize_global_layers(
            self,
            bias: bool,
            **kwargs
    ) -> None:
        """
        Initializes, if needed, the global layers.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            kwargs:
                Additional keyword arguments.
        """

    @abstractmethod
    def _initialize_local_layers(
            self,
            bias: bool,
            **kwargs
    ) -> None:
        """
        Initializes the global matrices.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            kwargs:
                Additional keyword arguments.
        """

    def forward(
            self,
            x: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass method.

        Args:
            x (torch.Tensor):
                Input tensor.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Output tensor.

        Raises:
            Exception:
                If the scope of the layer is different from 'global' or 'local'.
        """

        output = x

        for layer in self.structure:
            if layer["scope"] == "global":
                output = self.global_layers[layer["key"]].forward(output)
            elif layer["scope"] == "local":
                output = self.local_layers[layer["key"]].forward(output)
            else:
                raise Exception("The scope of the layer has to be 'global' or 'local'.")

        return output

    def reset_parameters(
            self
    ) -> None:
        """
        Resets the parameters of the layer.
        """

        for layer in self._local_layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    @property
    def shape(
            self
    ) -> torch.Size:
        """
        Returns the shape of the linear layer.

        Returns:
            torch.Size:
                Shape of the linear layer.
        """

        return torch.Size((self.in_features, self.out_features))

    @property
    def weight(
            self
    ) -> nn.Parameter:
        """
        Returns the weight parameter of the layer.

        Returns:
            Tensor:
                Weight parameter.
        """

        weight = None

        for layer in self.structure:
            if layer["scope"] == "global":
                if weight is None:
                    weight = self.global_layers[layer["key"]].weight
                else:
                    weight = torch.matmul(self.global_layers[layer["key"]].weight, weight)
            elif layer["scope"] == "local":
                if weight is None:
                    weight = self.local_layers[layer["key"]].weight
                else:
                    weight = torch.matmul(self.local_layers[layer["key"]].weight, weight)

        return weight

    @weight.setter
    def weight(
            self,
            value: torch.Tensor
    ) -> None:
        """
        Sets the weight parameter of the linear layer.

        For now, and probably forever, it is not implemented.

        Args:
            value (torch.Tensor):
                Weight parameter value.

        Raise:
            Exception:
                If the weight is tried to be set.
        """

        raise Exception('Cannot set the weight of a global dependent layer.')

    @property
    def bias(
            self
    ) -> nn.Parameter:
        """
        Returns the bias parameter of the linear layer.

        Returns:
            Tensor:
                Bias parameter.

        Raises:
            Exception:
                If the last layer has not the scope 'global' or 'local'.
        """

        if self.structure[-1]["scope"] == "global":
            return self.global_layers[self.structure[-1]["key"]].bias
        elif self.structure[-1]["scope"] == "local":
            return self.local_layers[self.structure[-1]["key"]].bias
        else:
            raise Exception("The last layer has to be global ('g') or local ('l').")

    @bias.setter
    def bias(
            self,
            value: torch.Tensor
    ) -> None:
        """
        Sets the bias parameter of the linear layer.

        Args:
            value (torch.Tensor):
                Bias parameter value.

        Raises:
            Exception:
                If the last layer has not the scope 'global' or 'local'.
        """

        if self.structure[-1]["scope"] == "global":
            if self.global_layers[self.structure[-1]["key"]].bias is not None:
                self.global_layers[self.structure[-1]["key"]].bias = value.detach().clone()
        elif self.structure[-1]["scope"] == "local":
            if self.local_layers[self.structure[-1]["key"]].bias is not None:
                self.local_layers[self.structure[-1]["key"]].bias = value.detach().clone()
        else:
            raise Exception("The last layer has to be global ('g') or local ('l').")

    @property
    def device(
            self
    ) -> torch.device:
        """
        Returns the device of the layer.

        Returns:
            torch.device:
                Device of the layer.
        """

        return next(self.parameters()).device

    
class GlobalDependentLinear(GlobalDependent):
    """
    Implementation of a Linear layer decomposed in the matrix product of many global and local matrices.
    It has a customizable layer structure based on the 'structure' attribute.

    Args:
        in_features (int):
            Number of input features.
        out_features (int):
            Number of output features.
        global_layers (nn.ModuleList):
            List of global matrices used in the linear layer.
        structure:
            Structure of the layer. Each element of the list has to contain a dictionary with the type of layer (
            'global' for global or 'local' for local), the shape of the layer and the key for the global matrix or the
            configuration for the local matrix.
        bias (bool):
            Flag to include bias in the linear layer.
        *args:
            Variable length argument list for defining input and output features.
        **kwargs:
            Arbitrary keyword arguments for layer configuration.

    Attributes:
        in_features (int):
            Number of input features.
        out_features (int):
            Number of output features.
        structure:
            Structure of the layer. Each element of the list has to contain a dictionary with the type of layer (
            'global' for global or 'local' for local), the shape of the layer and the key for the global matrix or the
            configuration for the local matrix.
        global_layers (dict):
            Dictionary containing the global layers.
        local_layers (nn.ModuleDict):
            Dictionary containing the local layers.

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            global_layers: nn.ModuleDict,
            structure,
            bias: bool = True,
            *args,
            **kwargs
    ) -> None:
        super(GlobalDependentLinear, self).__init__(in_features,
                                                    out_features,
                                                    global_layers,
                                                    structure,
                                                    bias,
                                                    *args,
                                                    **kwargs)

    def _create_layer(
            self,
            bias: bool,
            **kwargs
    ) -> None:
        """
        Creates the layer of the class based on the structure attribute.
        The layer is created as product of global and local matrices, and it is at the end a Linear layer.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            **kwargs:
                Arbitrary keyword arguments.
        """

        super()._create_layer(bias, **kwargs)

    def _initialize_global_layers(
            self,
            bias: bool = False,
            **kwargs
    ) -> None:
        """
        Initializes, if needed, the global layers as Linear layers.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            kwargs:
                Additional keyword arguments.
        """

        for layer in self.structure[:-1]:
            if layer["scope"] == "global":
                if layer["key"] not in self.global_layers.keys():
                    self.global_layers[layer["key"]] = nn.Linear(*layer["shape"], bias=False)

                    if "trainable" in layer.keys():
                        for param in self.global_layers[layer["key"]].parameters():
                            param.requires_grad = layer["trainable"]

        if self.structure[-1]["scope"] == "global":
            self.global_layers[self.structure[-1]["key"]] = nn.Linear(
                *self.structure[-1]["shape"],
                bias=True if bias else False
            )

            if "trainable" in self.structure[-1].keys():
                for param in self.global_layers[self.structure[-1]["key"]].parameters():
                    param.requires_grad = self.structure[-1]["trainable"]

    def _initialize_local_layers(
            self,
            bias: bool = False,
            **kwargs
    ) -> None:
        """
        Initializes, if needed, the local layers as Linear layers.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            kwargs:
                Additional keyword arguments.
        """

        self.local_layers = nn.ModuleDict()

        for layer in self.structure[:-1]:
            if layer["scope"] == "local":
                if layer["key"] not in self.local_layers.keys():
                    self.local_layers[layer["key"]] = nn.Linear(*layer["shape"], bias=False)

                    if "trainable" in layer.keys():
                        for param in self.local_layers[layer["key"]].parameters():
                            param.requires_grad = layer["trainable"]

        if self.structure[-1]["scope"] == "local":
            self.local_layers[self.structure[-1]["key"]] = nn.Linear(
                *self.structure[-1]["shape"],
                bias=True if bias else False
            )

            if "trainable" in self.structure[-1].keys():
                for param in self.local_layers[self.structure[-1]["key"]].parameters():
                    param.requires_grad = self.structure[-1]["trainable"]

    def merge(
            self,
            **kwargs
    ) -> nn.Module:
        """
        Merges the global and local layers into an equivalent linear layer.

        This method computes the equivalent linear layer by multiplying the weights
        of the global and local layers and setting the bias accordingly.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Equivalent linear layer with merged weights and bias.

        Raises:
            Exception:
                If the last layer's scope is neither 'global' nor 'local'.
        """

        equivalent_linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        weight = None

        for idx, layer in enumerate(self.structure):
            if layer["scope"] == "global":
                weight = self.global_layers[layer["key"]].weight.detach().clone() if weight is None else torch.matmul(
                    self.global_layers[layer["key"]].weight.detach().clone(),
                    weight
                )
            elif layer["scope"] == "local":
                weight = self.local_layers[
                    layer["key"]].weight.detach().clone() if weight is None else torch.matmul(
                    self.local_layers[layer["key"]].weight.detach().clone(),
                    weight
                )
            else:
                raise Exception("The last layer has to be global ('global') or local ('local').")

        with torch.no_grad():
            equivalent_linear.weight.data = weight

            if self.structure[-1]["scope"] == "global":
                equivalent_linear.bias.data = self.global_layers[self.structure[-1]["key"]].bias.detach().clone()
            elif self.structure[-1]["scope"] == "local":
                equivalent_linear.bias.data = self.local_layers[self.structure[-1]["key"]].bias.detach().clone()
            else:
                raise Exception("The last layer has to be global ('global') or local ('local').")

        return equivalent_linear


class StructureSpecificGlobalDependentLinear(ABC, nn.Module, MergeableLayer):
    """
    Abstract class that implements a linear layer with dependencies on global matrices that wraps a Linear layer.

    Args:
        target_layer (nn.Module):
            Target layer to be transformed in the factorized version.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        target_name (str):
            Name of the target layer.
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.

    Attributes:
        target_name (str):
            Name of the target layer.
        global_dependent_layer (GlobalDependentLinear):
            Linear layer with dependencies on global matrices.
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            target_name: str = None,
            *args,
            **kwargs
    ) -> None:
        super(StructureSpecificGlobalDependentLinear, self).__init__()

        self.target_name = target_name
        structure = self.define_structure(**{"target_layer": target_layer}, **kwargs)
        structure = self.post_process_structure(structure)

        self.global_dependent_layer = GlobalDependentLinear(
            target_layer.in_features,
            target_layer.out_features,
            global_layers,
            structure,
            target_layer.bias is not None
        )

        self.initialize_matrices(**{"target_layer": target_layer}, **kwargs)

    @abstractmethod
    def define_structure(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the structure of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.

        Returns:
            dict:
                Structure of the layer.
        """

    def post_process_structure(
            self,
            structure,
            **kwargs
    ) -> dict:
        """
        Post-processes the structure of the layer.

        Args:
            structure:
                Structure of the layer.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                Post-processed structure of the layer.
        """

        if self.target_name is not None:
            for idx, _ in enumerate(structure):
                structure[idx]["key"] = f"{self.target_name}_{structure[idx]['key']}"

        return structure

    def get_post_processed_key(
            self,
            key,
            **kwargs
    ) -> str:
        """
        Returns the post-processed key of the layer, given the key without post-processing (or with post-processing).

        Args:
            key (str):
                Key of the layer without post-processing.
            **kwargs:
                Additional keyword arguments.

        Returns:
            str:
                Post-processed key of the layer.
        """

        post_processed_key = key
        if self.target_name is not None and not key.startswith(self.target_name):
            post_processed_key = f"{self.target_name}_{key}"

        return post_processed_key

    def forward(
            self,
            x: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass method.

        Args:
            x (torch.Tensor):
                Input tensor.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Output tensor.
        """

        return self.global_dependent_layer(
            x,
            **kwargs
        )

    @abstractmethod
    def initialize_matrices(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the matrices of the layer.
        """

    def reset_parameters(
            self
    ) -> None:
        """
        Resets the parameters of the layer.
        """

        self.global_dependent_layer.reset_parameters()

    def merge(
            self,
            **kwargs
    ) -> nn.Module:
        """
        Merges the global and local layers into an equivalent linear layer.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Equivalent linear layer with merged weights and bias.
        """

        return self.global_dependent_layer.merge()

    def get_layer(
            self,
            scope: str,
            key: str,
            **kwargs
    ) -> nn.Module:
        """
        Returns the layer with the specified scope and key.

        Args:
            scope (str):
                Scope of the layer.
            key (str):
                Key of the layer.
            **kwargs:
                Additional keyword arguments.

        Returns:
            GlobalDependent: Layer with the specified scope and key.
        """

        if scope == "global":
            return self.global_dependent_layer.global_layers[self.get_post_processed_key(key)]
        elif scope == "local":
            return self.global_dependent_layer.local_layers[self.get_post_processed_key(key)]
        else:
            raise Exception("The scope of the layer has to be 'global' or 'local'.")

    def set_layer(
            self,
            scope: str,
            key: str,
            new_layer: nn.Module,
            **kwargs
    ) -> None:
        """
        Sets the layer with the specified scope and key.

        Args:
            scope (str):
                Scope of the layer.
            key (str):
                Key of the layer.
            new_layer (nn.Module):
                New layer to be set.
            **kwargs:
                Additional keyword arguments.
        """

        if scope == "global":
            self.global_dependent_layer.global_layers[self.get_post_processed_key(key)] = new_layer
        elif scope == "local":
            self.global_dependent_layer.local_layers[self.get_post_processed_key(key)] = new_layer
        else:
            raise Exception("The scope of the layer has to be 'global' or 'local'.")

    @property
    def structure(
            self
    ) -> dict:
        """
        Returns the structure of the layer.

        Returns:
            dict:
                Structure of the layer.
        """

        return self.global_dependent_layer.structure

    @property
    def shape(
            self
    ) -> torch.Size:
        """
        Returns the shape of the linear layer.

        Returns:
            torch.Size:
                Shape of the linear layer.
        """

        return self.global_dependent_layer.shape

    @property
    def weight(
            self
    ) -> nn.Parameter:
        """
        Returns the weight parameter of the layer.

        Returns:
            Tensor:
                Weight parameter.
        """

        return self.global_dependent_layer.weight

    @weight.setter
    def weight(
            self,
            value: torch.Tensor
    ) -> None:
        """
        SetS the weight parameter of the linear layer.

        For now, and probably forever, it is not implemented.

        Args:
            value (torch.Tensor):
                Weight parameter value.

        Raise:
            Exception:
                If the weight is tried to be set.
        """

        self.global_dependent_layer.weight = value

    @property
    def bias(
            self
    ) -> nn.Parameter:
        """
        Returns the bias parameter of the linear layer.

        Returns:
            Tensor:
                Bias parameter.
        """

        return self.global_dependent_layer.bias

    @bias.setter
    def bias(
            self,
            value: torch.Tensor
    ) -> None:
        """
        Property method to set the bias parameter of the linear layer.

        Args:
            value (torch.Tensor):
                Bias parameter value.
        """

        self.global_dependent_layer.bias = value


class LocalSVDLinear(StructureSpecificGlobalDependentLinear):
    """
    Implementation of a Linear layer on which is performed SVD decomposition.

    The layer is decomposed in two matrices:
    - a local matrix of shape (in_features, rank), that is the product between the matrix of the left singular vectors U
        and the matrix of the singular values S matrices of the truncated SVD;
    - a local matrix of shape (rank, out_features), that is the matrix of the right singular vectors V^T.

    Args:
        target_layer (nn.Module):
            Target layer to be transformed in the factorized version.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        rank (int):
            Rank of the global matrix.
        target_name (str):
            Name of the target layer.
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.

    Attributes:
        target_name (str):
            Name of the target layer.
        global_dependent_layer (GlobalDependentLinear):
            Linear layer with dependencies on global matrices.
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            rank: int,
            target_name: str = None,
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        super().__init__(target_layer, global_layers, target_name, *args, **kwargs)

    def define_structure(
            self,
            **kwargs
    ) -> Any:
        """
        Defines the structure of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]
        min_dim = min(target_layer.in_features, target_layer.out_features)

        return (
            {"scope": "local",
             "shape": (target_layer.in_features, min(min_dim, rank)),
             "key": "VT",
             "trainable": True},
            {"scope": "local",
             "shape": (min(min_dim, rank), target_layer.out_features),
             "key": "US",
             "trainable": True}
        )

    def initialize_matrices(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the matrices of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]

        U, S, VT = np.linalg.svd(target_layer.weight.data.numpy())
        min_dim = min(target_layer.in_features, target_layer.out_features)

        with torch.no_grad():
            self.get_layer("local", "US").weight.data = torch.tensor(
                U[:, :min(min_dim, rank)] @ np.diag(S[:min(min_dim, rank)])
            )
            self.get_layer("local", "VT").weight.data = torch.tensor(
                VT[:min(min_dim, rank), :]
            )

            for layer in self.structure:
                if "trainable" in layer.keys() and layer["scope"] == "local":
                    for params in self.get_layer("local", layer["key"]).parameters():
                        params.requires_grad = layer["trainable"]

            self.get_layer("local", "US").bias = target_layer.bias


class GlobalBaseLinear(StructureSpecificGlobalDependentLinear):
    """
    Implementation of a Linear layer on which is performed matrix decomposition in two matrices:
    - a global matrix of shape (in_features, rank);
    - a local matrix of shape (rank, out_features).

    Args:
        target_layer (nn.Module):
            Target layer to be transformed in the factorized version.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        rank (int):
            Rank of the global matrix.
        target_name (str):
            Name of the target layer.
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.

    Attributes:
        target_name (str):
            Name of the target layer.
        global_dependent_layer (GlobalDependentLinear):
            Linear layer with dependencies on global matrices.
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            rank: int,
            target_name: str = None,
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        super().__init__(
            target_layer,
            global_layers,
            target_name,
            *args,
            **kwargs
        )

    def define_structure(
            self,
            **kwargs
    ) -> Any:
        """
        Defines the structure of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]

        return (
            {"scope": "global",
             "shape": (target_layer.in_features, rank),
             "key": str((target_layer.in_features, rank)),
             "trainable": True},
            {"scope": "local",
             "shape": (rank, target_layer.out_features),
             "key": str((rank, target_layer.out_features)),
             "trainable": True}
        )

    def initialize_matrices(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the matrices of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        target_weight = target_layer.weight.data
        global_key = self.global_dependent_layer.structure[0]["key"]
        local_key = self.global_dependent_layer.structure[1]["key"]

        global_matrix = self.get_layer("global", global_key).weight.data

        with torch.no_grad():
            pinv_global_matrix = torch.pinverse(global_matrix)
            local_matrix = target_weight @ pinv_global_matrix

            self.get_layer("local", local_key).weight.data = local_matrix

            if "trainable" in self.structure[1].keys():
                for params in self.get_layer("local", local_key).parameters():
                    params.requires_grad = self.structure[1]["trainable"]

            self.get_layer("local", local_key).bias = target_layer.bias

        device = get_available_device()

        target_weight = target_weight.to(device)
        self.set_layer("global", global_key, self.get_layer("global", global_key).to(device))
        self.set_layer("local", local_key, self.get_layer("local", local_key).to(device))

        optimizer = torch.optim.AdamW([self.get_layer("local", local_key).weight])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )

        num_epochs = 50

        for epoch in range(num_epochs):
            loss = torch.norm((target_weight - torch.matmul(
                self.get_layer("local", local_key).weight,
                self.get_layer("global", global_key).weight)) ** 2)

            scheduler.step(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class GlobalFixedBaseLinear(GlobalBaseLinear):
    """
    Implementation of a Linear layer on which is performed matrix decomposition in two matrices:
    - a global non-trainable matrix of shape (in_features, rank);
    - a local matrix of shape (rank, out_features).

    The global matrix is fixed and not trainable.

    Args:
        target_layer (nn.Module):
            Target layer to be transformed in the factorized version.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        rank (int):
            Rank of the global matrix.
        target_name (str):
            Name of the target layer.
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.

    Attributes:
        target_name (str):
            Name of the target layer.
        global_dependent_layer (GlobalDependentLinear):
            Linear layer with dependencies on global matrices.
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            rank: int,
            target_name: str = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(target_layer, global_layers, rank, target_name, *args, **kwargs)

    def define_structure(
            self,
            **kwargs
    ) -> Any:
        """
        Defines the structure of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]

        return (
            {"scope": "global",
             "shape": (target_layer.in_features, rank),
             "key": str((target_layer.in_features, rank)),
             "trainable": False},
            {"scope": "local",
             "shape": (rank, target_layer.out_features),
             "key": str((rank, target_layer.out_features)),
             "trainable": True}
        )


# TODO: Implement the GlobalBaseSparseLinear class
class GlobalBaseSparseLinear(GlobalBaseLinear):
    """

    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            sparsity: float,
            target_name: str = None,
            *args,
            **kwargs
    ) -> None:
        self.sparsity = sparsity
        kwargs["sparsity"] = sparsity
        super().__init__(
            target_layer,
            global_layers,
            min(target_layer.in_features, target_layer.out_features),
            target_name,
            *args,
            **kwargs
        )

    def initialize_matrices(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the matrices of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        target_weight = target_layer.weight.data
        global_key = self.global_dependent_layer.structure[0]["key"]
        local_key = self.global_dependent_layer.structure[1]["key"]

        with torch.no_grad():
            global_matrix = self.get_layer("global", global_key).weight.data

            pinv_global_matrix = torch.pinverse(global_matrix)
            local_matrix = target_weight @ pinv_global_matrix

            self.get_layer("local", local_key).weight.data = torch.sparse_coo_tensor(
                indices=local_matrix.nonzero(),
                values=local_matrix[local_matrix.nonzero()].squeeze(),
                size=local_matrix.size()
            )

            if "trainable" in self.structure[1].keys():
                for params in self.get_layer("local", local_key).parameters():
                    params.requires_grad = self.structure[1]["trainable"]

            self.get_layer("local", local_key).bias = target_layer.bias

        optimizer = torch.optim.AdamW([self.get_layer("local", local_key).weight])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=False
        )

        num_epochs = 50

        for epoch in range(num_epochs):
            loss = torch.norm((target_weight - torch.matmul(
                self.get_layer("local", local_key).weight,
                self.get_layer("global", global_key).weight)) ** 2)

            scheduler.step(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class DiagonalLinearLayer(nn.Module):
    """
    A PyTorch module that implements a layer with diagonal weights.

    This layer behaves similar to `torch.nn.Linear` but with diagonal weights.
    It multiplies the input by diagonal weights and optionally adds a bias term.

    Args:
        in_features (int):
            Number of input features.
        out_features (int):
            Number of output features.
        bias (bool, optional):
            If True, adds a learnable bias to the output. Default is True.

    Attributes:
        weight (torch.Tensor):
            The learnable weights of the module. The shape is `(out_features, in_features)`.
        bias (torch.Tensor):
            The learnable bias of the module. If `bias` is set to False, this attribute is set to `None`.
    """

    def __init__(
            self,
            in_features,
            out_features,
            bias=True
    ) -> None:

        super(DiagonalLinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(
            self
    ) -> None:
        """
        Initializes the weights and biases with kaiming_uniform_ and uniform_ distributions, respectively.
        """

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
            self,
            x
    ) -> torch.Tensor:
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor):
                Input tensor of shape `(N, in_features)`.

        Returns:
            torch.Tensor:
                Output tensor of shape `(N, out_features)`.
        """

        output = x * self.weight

        if self.bias is not None:
            output += self.bias

        return output

    @property
    def weight(
            self,
            return_diag: bool = False
    ) -> nn.Parameter:
        """
        Returns the weight parameter of the layer.

        Args:
            return_diag (bool):
                If True, returns the diagonal weights. Default is False.

        Returns:
            Tensor: Weight parameter.
        """

        if return_diag:
            return self._weight
        else:
            return self._weight.diag()

    @weight.setter
    def weight(
            self,
            value: torch.Tensor
    ) -> None:
        """
        Property method to set the weight parameter of the linear layer.

        Args:
            value (torch.Tensor):
                Weight parameter value.

        Raises:
            ValueError:
                If the shape of the value is not `(out_features,)` or `(out_features, in_features)`.
        """
        if value.shape == (self.out_features,):
            self._weight = nn.Parameter(value)
        elif value.shape == (self.out_features, self.in_features):
            self._weight = nn.Parameter(torch.diag(value))
        else:
            raise ValueError("The shape of the weight should be (out_features,) or (out_features, in_features).")


# TODO: Implement the GlobalBaseDiagonalLinear class
class GlobalBaseDiagonalLinear(StructureSpecificGlobalDependentLinear):
    """
    Implementation of a Linear layer on which is performed matrix decomposition in two matrices:
    - a global matrix of shape (in_features, in_features);
    - a local diagonal matrix of shape (out_features, out_features).
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            target_name: str = None,
            *args,
            **kwargs
    ) -> None:
        """
        Initializes the layer.

        Args:
            target_layer (nn.Module):
                Target layer to be transformed in the factorized version.
            global_layers (nn.ModuleDict):
                Dictionary containing the global matrices.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.
        """

        super().__init__(target_layer, global_layers, target_name, *args, **kwargs)

    def define_structure(
            self,
            **kwargs
    ) -> Any:
        """
        Method to define the structure of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]

        return (
            {"scope": "global",
             "shape": (target_layer.in_features, self.rank),
             "key": str((target_layer.in_features, self.rank)),
             "trainable": False},
            {"scope": "local",
             "shape": (self.rank, target_layer.out_features),
             "key": str((self.rank, target_layer.out_features)),
             "trainable": True}
        )

    def initialize_matrices(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the matrices of the layer.
        """

        target_layer = kwargs["target_layer"]
        target_weight = target_layer.weight.data
        global_key = self.global_dependent_layer.structure[0]["key"]
        local_key = self.global_dependent_layer.structure[1]["key"]

        with torch.no_grad():
            global_matrix = self.global_dependent_layer.global_layers[global_key].weight.data

            pinv_global_matrix = torch.pinverse(global_matrix)
            local_matrix = target_weight @ pinv_global_matrix

            self.global_dependent_layer.local_layers[local_key].weight.data = local_matrix

            if "trainable" in self.global_dependent_layer.structure[1].keys():
                for params in self.global_dependent_layer.local_layers[local_key].parameters():
                    params.requires_grad = self.global_dependent_layer.structure[1]["trainable"]

            self.global_dependent_layer.local_layers[local_key].bias = target_layer.bias

        optimizer = torch.optim.AdamW([self.global_dependent_layer.local_layers[local_key].weight])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=False
        )

        num_epochs = 50

        for epoch in range(num_epochs):
            loss = torch.norm((target_weight - torch.matmul(
                self.global_dependent_layer.local_layers[local_key].weight,
                self.global_dependent_layer.global_layers[global_key].weight)) ** 2)

            scheduler.step(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    import time
    linear_layer = nn.Linear(10000, 10000, bias=True)
    rank = 100

    init_time = time.time()

    global_matrices_dict = nn.ModuleDict()
    gbl = GlobalBaseLinear(
        linear_layer,
        global_matrices_dict,
        rank=rank,
        target_name="query",
    )

    print(f"Time taken: {(time.time() - init_time)}")
    
    print(gbl)

    #gbl_merged = gbl.merge()

    """
    print("Weights")
    print("Weights of the original layer")
    print(linear_layer.weight.shape)
    print()
    print("Weights of the global dependent layer")
    print(gbl_merged.weight)
    print()
    """
    """
    print(gbl)
    print(gbl_merged)
    print(gbl_merged.weight.data - linear_layer.weight.data)
    tolerance = 1e-7
    assert torch.allclose(gbl_merged.weight.data, linear_layer.weight.data, atol=tolerance)
    """
    """
    print("Output example")
    x = torch.ones(100, 100)
    print("Output of the original layer")
    print(linear_layer(x))
    print()
    print("Output of the global dependent layer")
    print(gbl(x))
    """

