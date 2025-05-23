from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

import numpy as np
import math

from typing_extensions import override

from exporch import get_available_device

from imports.HadamardDecomposition.alternating_gradient_descent import \
    scaled_alternating_gradient_descent_hadDec
from imports.HadamardDecomposition.minibatch_stochastic_gradient_descent import \
    mb_stochastic_gradient_descent_hadDec

from neuroflex.factorization.layers.factorized_layer import GlobalDependent, StructureSpecificGlobalDependent, FactorizedLayer


class FactorizedLinearLayer(FactorizedLayer, ABC):
    """
    Abstract class that implements a linear layer on which is performed matrix decomposition.
    """

    @override
    def compute_approximation_stats(
            self,
            target_layer: torch.nn.Linear
    ) -> dict:
        """
        Computes the approximation statistics of the layer.

        Args:
            target_layer (torch.nn.Linear):
                Target layer to be approximated.

        Returns:
            dict:
                Approximation statistics.
        """

        approximation_stats = {
            "MSE approximation": (target_layer.weight.data.to(torch.float32) - self.weight.data.to(torch.float32)).pow(2).mean().item(),
        }

        return approximation_stats


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
            dtype: torch.dtype = None,
            *args,
            **kwargs
    ) -> None:
        kwargs["bias"] = bias
        super().__init__(
            in_features,
            out_features,
            global_layers,
            structure,
            dtype,
            *args,
            **kwargs
        )

    def _create_layer(
            self,
            **kwargs
    ) -> None:
        """
        Creates the layer of the class based on the structure attribute.
        The layer is created as product of global and local matrices, and it is at the end a Linear layer.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        super()._create_layer(**kwargs)

    def _initialize_global_layers(
            self,
            **kwargs
    ) -> None:
        """
        Initializes, if needed, the global layers as Linear layers.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        bias = kwargs["bias"]
        global_initialization = kwargs["global_initialization"] if "global_initialization" in kwargs.keys() else "random"

        for layer in self.structure[:-1]:
            if layer["scope"] == "global":
                if layer["key"] not in self.global_layers.keys():
                    self.global_layers[layer["key"]] = nn.Linear(
                        *layer["shape"],
                        bias=False,
                        dtype=self.dtype
                    )

                    if global_initialization == "identity":
                        in_features = self.global_layers[layer["key"]].in_features
                        out_features = self.global_layers[layer["key"]].out_features
                        with torch.no_grad():
                            diag_size = min(in_features, out_features)
                            diagonal_matrix = torch.zeros(out_features, in_features)
                            if in_features > out_features:
                                diagonal_matrix[:, out_features:] = torch.ones(out_features, in_features - out_features) / torch.sqrt(out_features)
                            else:
                                diagonal_matrix[in_features:, :] = torch.ones(out_features - in_features, in_features) / torch.sqrt(in_features)
                            diagonal_matrix[range(diag_size), range(diag_size)] = torch.ones(diag_size)
                            self.global_layers[layer["key"]].weight.data = diagonal_matrix.to(self.dtype)

                    if "trainable" in layer.keys():
                        for param in self.global_layers[layer["key"]].parameters():
                            param.requires_grad = layer["trainable"]

        if self.structure[-1]["scope"] == "global":
            self.global_layers[self.structure[-1]["key"]] = nn.Linear(
                *self.structure[-1]["shape"],
                bias=True if bias else False,
                dtype=self.dtype
            )

            if global_initialization == "identity":
                in_features = self.global_layers[self.structure[-1]["key"]].in_features
                out_features = self.global_layers[self.structure[-1]["key"]].out_features
                with torch.no_grad():
                    diag_size = min(in_features, out_features)
                    diagonal_matrix = torch.zeros(out_features, in_features)
                    diagonal_matrix[range(diag_size), range(diag_size)] = torch.randn(diag_size)
                    self.global_layers[self.structure[-1]["key"]].weight.data = diagonal_matrix.to(self.dtype)

            if "trainable" in self.structure[-1].keys():
                for param in self.global_layers[self.structure[-1]["key"]].parameters():
                    param.requires_grad = self.structure[-1]["trainable"]

    def _initialize_local_layers(
            self,
            **kwargs
    ) -> None:
        """
        Initializes, if needed, the local layers as Linear layers.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        bias = kwargs["bias"]
        self.local_layers = nn.ModuleDict()

        for layer in self.structure[:-1]:
            if layer["scope"] == "local":
                if layer["key"] not in self.local_layers.keys():
                    self.local_layers[layer["key"]] = nn.Linear(
                        *layer["shape"],
                        bias=False,
                        dtype=self.dtype
                    )

                    if "trainable" in layer.keys():
                        for param in self.local_layers[layer["key"]].parameters():
                            param.requires_grad = layer["trainable"]

        if self.structure[-1]["scope"] == "local":
            self.local_layers[self.structure[-1]["key"]] = nn.Linear(
                *self.structure[-1]["shape"],
                bias=True if bias else False,
                dtype=self.dtype
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


class StructureSpecificGlobalDependentLinear(StructureSpecificGlobalDependent, ABC):
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
            average_layers: nn.ModuleDict,
            target_name: str = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            target_layer,
            global_layers,
            average_layers,
            target_name,
            *args,
            **kwargs
        )

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

    def define_global_dependent_layer(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            structure: dict,
            **kwargs
    ) -> GlobalDependent:
        """
        Defines the global dependent layer.

        Args:
            target_layer (nn.Module):
                Target layer to be transformed in the factorized version.
            global_layers (nn.ModuleDict):
                Dictionary containing the global matrices.
            structure (dict):
                Structure of the layer.
            **kwargs:
                Additional keyword arguments.

        Returns:
            GlobalDependent:
                Global Linear dependent layer.
        """

        return GlobalDependentLinear(
            target_layer.in_features,
            target_layer.out_features,
            global_layers,
            structure,
            target_layer.bias is not None,
            dtype=target_layer.weight.dtype,
            **kwargs
        )

    def define_average_matrix_layer(
            self,
            average_matrix: torch.Tensor,
            **kwargs
    ) -> nn.Module:
        """
        Defines a Linear layer that will be the one used as layer containing the average matrix of some grouped
        layers.

        Args:
            average_matrix (torch.Tensor):
                Average matrix.
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Linear layer containing the average matrix.
        """

        layer = nn.Linear(
            average_matrix.size(0),
            average_matrix.size(1),
            bias=False
        )
        with torch.no_grad():
            layer.weight.data = average_matrix

        return layer

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
            Additional keyword arguments.

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
            average_layers: nn.ModuleDict,
            target_name: str,
            rank: int,
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        super().__init__(
            target_layer,
            global_layers,
            average_layers,
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
        dtype = target_layer.weight.data.dtype
        rank = kwargs["rank"]

        U, S, VT = np.linalg.svd(target_layer.weight.data.to(torch.float32).numpy())
        min_dim = min(target_layer.in_features, target_layer.out_features)

        with torch.no_grad():
            self.get_layer("local", "US").weight.data = torch.tensor(
                U[:, :min(min_dim, rank)] @ np.diag(S[:min(min_dim, rank)])
            ).to(dtype)

            self.get_layer("local", "VT").weight.data = torch.tensor(VT[:min(min_dim, rank), :]).to(dtype)

            for layer in self.structure:
                if "trainable" in layer.keys() and layer["scope"] == "local":
                    for params in self.get_layer("local", layer["key"]).parameters():
                        params.requires_grad = layer["trainable"]

            self.get_layer("local", "US").bias = target_layer.bias


# TODO here the Hadamard decomposition cannot be trained since the matrices A_i are multiplied and the matrices B_i are multiplied
# TODO so only equal ranks

# TODO IT IS WRONG
class LocalHadamardLinear(StructureSpecificGlobalDependentLinear):
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
            Additional keyword arguments.

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
            average_layers: nn.ModuleDict,
            target_name: str,
            rank: int,
            method: str = "alternating gradient",
            learning_rate: float = 0.01,
            max_iterations: int = 1000,
            *args,
            **kwargs
    ) -> None:
        kwargs.update({
            "rank": rank,
            "method": method,
            "learning_rate": learning_rate,
            "max_iterations": max_iterations
        })
        super().__init__(
            target_layer,
            global_layers,
            average_layers,
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
        min_dim = min(target_layer.in_features, target_layer.out_features)

        return (
            {"scope": "local",
             "shape": (target_layer.in_features, min(min_dim, rank)),
             "key": "B",
             "trainable": True},
            {"scope": "local",
             "shape": (min(min_dim, rank), target_layer.out_features),
             "key": "A",
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
        dtype = target_layer.weight.data.dtype
        device = target_layer.weight.data.device
        rank = kwargs["rank"]
        method = kwargs["method"]
        learning_rate = kwargs["learning_rate"]
        max_iterations = kwargs["max_iterations"]

        _, _, [A1, B1, A2, B2], _, _ = scaled_alternating_gradient_descent_hadDec(
            target_layer.weight.data.cpu().to(torch.float32).numpy(), rank, learning_rate, max_iterations
        ) if method == "alternating gradient" else mb_stochastic_gradient_descent_hadDec(
            target_layer.weight.data.cpu().to(torch.float32).numpy(), rank, learning_rate, max_iterations
        )

        with torch.no_grad():
            self.get_layer("local", "A").weight.data = torch.tensor(A1 * A2).to(dtype).to(device)
            self.get_layer("local", "B").weight.data = torch.tensor(B1 * B2).to(dtype).to(device)

            for layer in self.structure:
                if "trainable" in layer.keys() and layer["scope"] == "local":
                    for params in self.get_layer("local", layer["key"]).parameters():
                        params.requires_grad = layer["trainable"]

            self.get_layer("local", "A").bias = target_layer.bias


class HadamardLinearLayer(FactorizedLinearLayer):
    """
    Implementation of a Linear layer on which is performed Hadamard decomposition.

    The layer is decomposed in four matrices:
        -
        -
        -
        -
    """

    def __init__(
            self,
            target_layer: torch.nn.Linear,
            target_name: str,
            rank: int,
            learning_rate: float,
            max_iterations: int,
            **kwargs
    ) -> None:
        kwargs.update({
            "rank": rank,
            "learning_rate": learning_rate,
            "max_iterations": max_iterations
        })
        super().__init__(
            target_layer,
            target_name,
            **kwargs
        )

        self.rank = rank
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    @override
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor):
                Input tensor.

        Returns:
            torch.Tensor:
                Output tensor.
        """

        A_1 = self.factorized_layer["A_1"]
        A_2 = self.factorized_layer["A_2"]
        B_1 = self.factorized_layer["B_1"]
        B_2 = self.factorized_layer["B_2"]
        bias = self.factorized_layer["bias"]

        y = torch.matmul(A_1, torch.matmul(B_1, x)) * torch.matmul(A_2, torch.matmul(B_2, x))
        if bias is not None:
            y += bias

        return y

    @override
    def factorize_layer(
            self,
            target_layer,
            **kwargs
    ) -> dict:
        """
        Factorizes the target layer.

        Args:
            target_layer:
                Target layer to be factorized.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                Factorized layer.
        """

        # Getting the parameters
        rank = kwargs["rank"]
        learning_rate = kwargs["learning_rate"]
        max_iterations = kwargs["max_iterations"]
        dtype = target_layer.weight.data.dtype
        device = target_layer.weight.data.device

        _, _, [A_1, B_1, A_2, B_2], _, _ = scaled_alternating_gradient_descent_hadDec(
            target_layer.weight.data.cpu().to(torch.float32).numpy(), rank, learning_rate, max_iterations
        )

        factorized_layer = {
            "A_1": torch.nn.Parameter(torch.tensor(A_1).to(dtype).to(device)),
            "A_2": torch.nn.Parameter(torch.tensor(A_2).to(dtype).to(device)),
            "B_1": torch.nn.Parameter(torch.tensor(B_1).to(dtype).to(device)),
            "B_2": torch.nn.Parameter(torch.tensor(B_2).to(dtype).to(device)),
            "bias": target_layer.bias if target_layer.bias is not None else None
        }

        return factorized_layer

    @property
    def weight(
            self
    ) -> torch.Tensor:
        """
        Returns the weight parameter of the layer.

        Returns:
            torch.Tensor:
                Weight parameter.
        """

        return torch.matmul(self.factorized_layer["A_1"], self.factorized_layer["B_1"]) * torch.matmul(self.factorized_layer["A_2"], self.factorized_layer["B_2"])

    @property
    def bias(
            self
    ) -> torch.Tensor:
        """
        Returns the bias parameter of the layer.

        Returns:
            torch.Tensor:
                Bias parameter.
        """

        return self.factorized_layer["bias"]


if __name__ == "__main__":
    dim = 4000
    target_layer = nn.Linear(dim, dim)
    rank = dim
    learning_rate = 0.01
    max_iterations = 10000

    hadamard_linear_layer = HadamardLinearLayer(target_layer, "Hadamard Linear Layer", rank, learning_rate, max_iterations)

    print(target_layer.weight)
    print(hadamard_linear_layer.weight)
    print(target_layer.bias)
    print(hadamard_linear_layer.bias)

    print(hadamard_linear_layer.approximation_stats)


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
            Additional keyword arguments.

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
            average_layers: nn.ModuleDict,
            target_name: str,
            rank: int,
            initialization_type: str = "pseudo-inverse",
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        kwargs["initialization_type"] = initialization_type
        kwargs["global_initialization"] = "random"
        super().__init__(
            target_layer,
            global_layers,
            average_layers,
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
             "key": f"({target_layer.in_features},{rank})",
             "trainable": True},
            {"scope": "local",
             "shape": (rank, target_layer.out_features),
             "key": f"({rank},{target_layer.out_features})",
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
        initialization_type = kwargs["initialization_type"]

        global_matrix = self.get_layer("global", global_key).weight.data

        with torch.no_grad():
            if initialization_type == "pseudo-inverse":
                global_matrix = global_matrix.to(torch.float32)

                pinv_global_matrix = torch.pinverse(global_matrix.to("cpu"))

                local_matrix = target_weight.to(torch.float32) @ pinv_global_matrix.to(torch.float32)
                local_matrix = local_matrix.to(self.global_dependent_layer.dtype)

                self.get_layer("local", local_key).weight.data = local_matrix
                self.get_layer("local", local_key).bias = target_layer.bias
            elif initialization_type == "random":
                pass
            else:
                raise ValueError("Initialization type not recognized.")

        if "trainable" in self.structure[1].keys():
            for params in self.get_layer("local", local_key).parameters():
                params.requires_grad = self.structure[1]["trainable"]

        if initialization_type in ["random", ]:
            device = get_available_device()

            # Getting the tensors and moving them to the device
            target_weight = target_weight.to(device)

            local_weight = self.get_layer("local", local_key).weight.data.clone().to(device)
            local_weight = nn.Parameter(local_weight, requires_grad=True)
            local_weight = local_weight.to(device)

            global_weight = self.get_layer("global", global_key).weight.data.to(device)

            # Configuring the optimizer
            eps = 1e-7 if target_weight.dtype == torch.float16 else 1e-8
            optimizer = torch.optim.AdamW([local_weight], lr=1e-2, eps=eps)

            # Configuring the loss
            loss_fn = torch.nn.MSELoss()

            num_epochs = 1000
            for epoch in range(num_epochs):
                # Forward pass
                loss = loss_fn(torch.matmul(local_weight, global_weight).to(device), target_weight.to(device))

                # Backward pass
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
            Additional keyword arguments.

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
            average_layers: nn.ModuleDict,
            target_name: str,
            rank: int,
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        super().__init__(
            target_layer,
            global_layers,
            average_layers,
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
             "trainable": False},
            {"scope": "local",
             "shape": (rank, target_layer.out_features),
             "key": str((rank, target_layer.out_features)),
             "trainable": True}
        )


class GLAMSVDLinear(StructureSpecificGlobalDependentLinear):
    """
    Implementation of a Linear layer on which is performed matrix decomposition in two matrices:
    - a global matrix of shape (in_features, rank);
    - a local matrix of shape (rank, out_features).

    The matrices are initialized with the SVD decomposition of the target layer.
    Global matrices will be trained trying to minimize their absolute difference in order to be able to prune some of
    them.

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
            Additional keyword arguments.

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
            average_layers: nn.ModuleDict,
            target_name: str,
            rank: int,
            *args,
            **kwargs
    ) -> None:
        kwargs["rank"] = rank
        kwargs["path"] = f"{kwargs['path']}_{target_name}"

        super().__init__(
            target_layer,
            global_layers,
            average_layers,
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
        path = kwargs["path"]

        return (
            {"scope": "global",
             "shape": (target_layer.in_features, rank),
             "key": f"({target_layer.in_features},{rank})_{path}",
             "trainable": True},
            {"scope": "local",
             "shape": (rank, target_layer.out_features),
             "key": f"({rank}, {target_layer.out_features})",
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
        dtype = target_layer.weight.data.numpy().dtype
        rank = kwargs["rank"]

        global_key = self.global_dependent_layer.structure[0]["key"]
        local_key = self.global_dependent_layer.structure[1]["key"]

        U, S, VT = np.linalg.svd(target_layer.weight.data.numpy().astype(np.float32))
        min_dim = min(target_layer.in_features, target_layer.out_features)

        with torch.no_grad():
            self.get_layer("local", local_key).weight.data = torch.tensor(
                U[:, :min(min_dim, rank)].astype(dtype)
            ) @ np.diag(
                S[:min(min_dim, rank)].astype(dtype)
            )

            self.get_layer("global", global_key).weight.data = torch.tensor(
                VT[:min(min_dim, rank), :].astype(dtype)
            )

            for layer in self.structure:
                if "trainable" in layer.keys() and layer["scope"] == "local":
                    for params in self.get_layer("local", layer["key"]).parameters():
                        params.requires_grad = layer["trainable"]

            self.get_layer("local", local_key).bias = target_layer.bias


# TODO: Implement the GlobalBaseSparseLinear class
class GlobalBaseSparseLinear(GlobalBaseLinear):
    """

    """

    def __init__(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            average_layers: nn.ModuleDict,
            target_name: str,
            sparsity: float,
            *args,
            **kwargs
    ) -> None:
        self.sparsity = sparsity
        kwargs["sparsity"] = sparsity
        super().__init__(
            target_layer,
            global_layers,
            average_layers,
            target_name,
            min(target_layer.in_features, target_layer.out_features),
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
            return torch.Parameter(self._weight.diag())

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
            average_layers: nn.ModuleDict,
            target_name: str,
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

        super().__init__(
            target_layer,
            global_layers,
            average_layers,
            target_name,
            *args,
            **kwargs
        )

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
