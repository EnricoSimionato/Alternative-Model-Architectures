from typing import Any

from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn


class GlobalDependent(ABC, nn.Module):
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
        structure:
            Structure of the layer. Each element of the list has to contain a tuple with the type of layer ('g' for
            global or 'l' for local) and the key for the global matrix or the configuration for the local matrix.
        bias (bool):
            Flag to include bias in the linear layer.
        *args:
            Variable length argument list for defining input and output features.
        **kwargs:
            Arbitrary keyword arguments for layer configuration.
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
            torch.Tensor: Output tensor.

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
            torch.Size: Shape of the linear layer.
        """

        return torch.Size((self.in_features, self.out_features))

    @property
    def weight(
            self
    ) -> nn.Parameter:
        """
        Returns the weight parameter of the layer.

        Returns:
            Tensor: Weight parameter.
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
        Property method to set the weight parameter of the linear layer.

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
        Property method to obtain the bias parameter of the linear layer.

        Returns:
            Tensor: Bias parameter.

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
        Property method to set the bias parameter of the linear layer.

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


class GlobalDependentLinear(GlobalDependent):
    """
    Implementation of a Linear layer decomposed in the matrix product of many global and local matrices.
    It has a customizable layer structure based on the 'structure' attribute.

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


class StructureSpecificGlobalDependentLinear(ABC, nn.Module):
    """
    Abstract class that implements a linear layer with dependencies on global matrices that wraps a Linear layer.
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            *args,
            **kwargs
    ) -> None:
        super(StructureSpecificGlobalDependentLinear, self).__init__()

        structure = self.define_structure(**{"target_layer": target_layer}, **kwargs)
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
    ) -> Any:
        """
        Method to define the structure of the layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
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
            torch.Tensor: Output tensor.
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

    @property
    def shape(
            self
    ) -> torch.Size:
        """
        Returns the shape of the linear layer.

        Returns:
            torch.Size: Shape of the linear layer.
        """

        return self.global_dependent_layer.shape

    @property
    def weight(
            self
    ) -> nn.Parameter:
        """
        Returns the weight parameter of the layer.

        Returns:
            Tensor: Weight parameter.
        """

        return self.global_dependent_layer.weight

    @weight.setter
    def weight(
            self,
            value: torch.Tensor
    ) -> None:
        """
        Property method to set the weight parameter of the linear layer.

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
        Property method to obtain the bias parameter of the linear layer.

        Returns:
            Tensor: Bias parameter.
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


class GlobalBaseLinear(StructureSpecificGlobalDependentLinear):
    """
    Implementation of a Linear layer on which is performed matrix decomposition in two matrices:
    - a global matrix of shape (in_features, rank);
    - a local matrix of shape (rank, out_features).
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            rank: int,
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        super().__init__(target_layer, global_layers, *args, **kwargs)

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


class LocalSVDLinear(StructureSpecificGlobalDependentLinear):
    """
    Implementation of a Linear layer on which is performed SVD decomposition.
    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            rank: int,
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        super().__init__(target_layer, global_layers, *args, **kwargs)

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
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]

        U, S, VT = np.linalg.svd(target_layer.weight.data.numpy())
        min_dim = min(target_layer.in_features, target_layer.out_features)

        with torch.no_grad():
            self.global_dependent_layer.local_layers["US"].weight.data = torch.tensor(
                U[:, :min(min_dim, rank)] @ np.diag(S[:min(min_dim, rank)])
            )
            self.global_dependent_layer.local_layers["VT"].weight.data = torch.tensor(
                VT[:min(min_dim, rank), :]
            )

            for layer in self.global_dependent_layer.structure:
                if "trainable" in layer.keys() and layer["scope"] == "local":
                    for params in self.global_dependent_layer.local_layers[layer["key"]].parameters():
                        params.requires_grad = layer["trainable"]

            self.global_dependent_layer.local_layers["US"].bias = target_layer.bias


class GlobalFixedRandomBaseLinear(StructureSpecificGlobalDependentLinear):
    """
    Implementation of a Linear layer on which is performed matrix decomposition in two matrices:
    - a global matrix of shape (in_features, rank);
    - a local matrix of shape (rank, out_features).


    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            rank: int,
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
            rank (int):
                Rank of the global matrix.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.
        """
        super().__init__(target_layer, global_layers, rank, *args, **kwargs)

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
            self
    ) -> None:
        """
        Initializes the matrices of the layer.
        """

        with torch.no_grad():
            self.global_dependent_layer.global_matrices[self.global_dependent_layer.structure[0]["key"]].weight.data.copy_(
                torch.randn(self.rank, self.global_dependent_layer.structure[0]["shape"][0])
            )
            self.global_dependent_layer.global_matrices[self.global_dependent_layer.structure[1]["key"]].weight.data.copy_(
                torch.randn(self.global_dependent_layer.structure[1]["shape"][0], self.rank)
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
    linear_layer = nn.Linear(100, 100, bias=True)
    global_matrices_dict = nn.ModuleDict()
    gbl = GlobalBaseLinear(
        linear_layer,
        global_matrices_dict,
        100
    )

    print("Weights")
    print("Weights of the original layer")
    print(linear_layer.weight)
    print()
    print("Weights of the global dependent layer")
    print(gbl.weight)
    print()

    print("Output example")
    x = torch.ones(100, 100)
    print("Output of the original layer")
    print(linear_layer(x))
    print()
    print("Output of the global dependent layer")
    print(gbl(x))