from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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


class GlobalDependent(nn.Module, MergeableLayer, ABC):
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
            Additional keyword arguments.

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
            dtype: torch.dtype = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.global_layers = global_layers
        self.local_layers = nn.ModuleDict()
        self.dtype = dtype

        # Initial sanity check on the structure of the layer
        if len(structure) < 1:
            raise Exception("The structure has to contain at least one layer")
        for layer in structure:
            if layer["scope"] != "global" and layer["scope"] != "local":
                raise Exception("The structure of each layer has to contain 'global' or 'local' as 'scope'")
        self.structure = structure

        self._create_layer(**kwargs)

    def _create_layer(
            self,
            **kwargs
    ) -> None:
        """
        Creates the actual layer of the class.

        Args:
            **kwargs:
                Arbitrary keyword arguments.
        """

        self._initialize_global_layers(**kwargs)
        self._initialize_local_layers(**kwargs)

    @abstractmethod
    def _initialize_global_layers(
            self,
            **kwargs
    ) -> None:
        """
        Initializes, if needed, the global layers.

        Args:
            kwargs:
                Additional keyword arguments.
        """

    @abstractmethod
    def _initialize_local_layers(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the global matrices.

        Args:
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


class StructureSpecificGlobalDependent(nn.Module, MergeableLayer, ABC):
    """
    Abstract class that implements a layer with dependencies on global matrices that wraps a target layer.

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
        average_matrices_layer (nn.ModuleDict):
            Layer that applies the average matrix to the input.
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
        super().__init__()

        self.target_name = target_name
        structure = self._define_structure(**{"target_layer": target_layer}, **kwargs)

        average_matrix = None if "average_matrix" not in kwargs else kwargs["average_matrix"]
        self.average_matrix_layer = None if average_matrix is None else average_layers

        self.global_dependent_layer = self._define_global_dependent_layer(
            target_layer,
            global_layers,
            structure,
            **kwargs
        )

        self._define_average_matrix_layer(
            average_matrix,
        )

        self.initialize_matrices(**{"target_layer": target_layer}, **kwargs)

    def _define_structure(
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

        structure = self.define_structure(**kwargs)

        # Post-processing of the structure
        if self.target_name is not None:
            for idx, _ in enumerate(structure):
                structure[idx]["key"] = f"{self.target_name}_{structure[idx]['key']}"

        return structure

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

    def _define_global_dependent_layer(
            self,
            target_layer: nn.Module,
            global_layers: nn.ModuleDict,
            structure: dict,
            average_matrix: torch.Tensor = None,
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
                Global dependent layer.
        """

        if average_matrix is not None:
            with torch.no_grad():
                target_layer.weight.data = target_layer.weight.data - average_matrix.data

        return self.define_global_dependent_layer(
            target_layer,
            global_layers,
            structure,
            **kwargs
        )

    @abstractmethod
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
                Global dependent layer.
        """

    def _define_average_matrix_layer(
            self,
            average_matrix: torch.Tensor,
            **kwargs
    ) -> None:
        """
        Defines the average matrix layer that is the layer that applies the average matrix to the input following the
        specific strategy of the class of the layer.

        Args:
            average_matrix (torch.Tensor):
                Average matrix.
            **kwargs:
                Additional keyword arguments.
        """

        if self.average_matrix_layer is not None:
            if self.get_average_matrix_key() not in self.average_matrix_layer.keys():
                average_matrix_layer = self.define_average_matrix_layer(
                        average_matrix,
                        **kwargs
                )

                for param in average_matrix_layer.parameters():
                    param.requires_grad = False

                self.average_matrix_layer[self.get_average_matrix_key()] = average_matrix_layer

    def define_average_matrix_layer(
            self,
            average_matrix: torch.Tensor,
            **kwargs
    ) -> nn.Module:
        """
        Defines the average matrix layer that is the layer that applies the average matrix to the input following the
        specific strategy of the class of the layer.

        Args:
            average_matrix (torch.Tensor):
                Average matrix.
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Average matrix layer.
        """

        return None

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

        output = self.global_dependent_layer(
            x,
            **kwargs
        )
        if self.average_matrix_layer is not None:
            return output + self.get_average_matrix().forward(x)
        else:
            return output

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

    def change_key(
            self,
            scope: str,
            previous_key: str,
            new_key: str,
            **kwargs
    ) -> None:
        """
        Changes the key of a layer given the previous key and the new one.

        Args:
            scope (str):
                Scope of the layer.
            previous_key (str):
                Previous key of the layer.
            new_key (str):
                New key of the layer.
            **kwargs:
                Additional keyword arguments.
        """

        changed = False
        for idx, layer in enumerate(self.structure):
            if layer["scope"] == scope and layer["key"] == previous_key:
                self.structure[idx]["key"] = self.get_post_processed_key(new_key, **kwargs)
                changed = True
                break

        if not changed:
            print(f"The layer with the scope '{scope}' and key '{previous_key}' does not exist.")
        else:
            print(f"The layer with the scope '{scope}' and key '{previous_key}' has now key '{new_key}'.")

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

    def get_average_matrix_key(
            self,
            **kwargs
    ) -> str:
        """
        Returns the key of the average matrix.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            str:
                Key of the average matrix.
        """

        return f"{self.target_name}_({self.global_dependent_layer.shape[0]},{self.global_dependent_layer.shape[1]})"

    def get_average_matrix(
            self,
            **kwargs
    ) -> nn.Module:
        """
        Returns the average matrix layer.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Average matrix layer.
        """

        return self.average_matrix_layer[self.get_average_matrix_key()]

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
