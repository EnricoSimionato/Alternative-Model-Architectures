from abc import ABC, abstractmethod
from typing import Any

import numpy as np

import torch
import torch.nn as nn

from exporch import get_available_device
from neuroflex.layers.factorized_layer import GlobalDependent, StructureSpecificGlobalDependent


class GlobalDependentEmbedding(GlobalDependent):
    """
    Implementation of an Embedding layer decomposed in the matrix product of many global and local matrices.
    It has a customizable layer structure based on the 'structure' attribute.

    Args:
        num_embeddings (int):
            Number of embeddings.
        embeddings_dim (int):
            Dimension of the embeddings.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        structure:
            Structure of the layer. Each element of the list has to contain a dictionary with the type of layer (
            'global' for global or 'local' for local), the shape of the layer and the key for the global matrix or the
            configuration for the local matrix.
        padding_idx (int):
            Padding index.
        max_norm (float):
            Maximum norm of the embeddings.
        norm_type (float):
            Norm type.
        scale_grad_by_freq (bool):
            Flag to scale the gradients by frequency.
        sparse (bool):
            Flag to use sparse gradients.
        _weight (torch.Tensor):
            Weight tensor.
        _freeze (bool):
            Flag to freeze the layer.
        device (torch.device):
            Device of the layer.
        dtype (torch.dtype):
            Data type of the layer.
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.


    Attributes:
        in_features (int):
            Number of input features.
        out_features (int):
            Number of output features.
        global_layers (dict):
            Dictionary containing the global layers.
        local_layers (nn.ModuleDict):
            Dictionary containing the local layers.
        structure:
            Structure of the layer. Each element of the list has to contain a dictionary with the type of layer (
            'global' for global or 'local' for local), the shape of the layer and the key for the global matrix or the
            configuration for the local matrix.
    """

    def __init__(
            self,
            num_embeddings: int,
            embeddings_dim: int,
            global_layers: nn.ModuleDict,
            structure: dict,
            padding_idx: int = None,
            max_norm: float = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
            _weight: torch.Tensor = None,
            _freeze: bool = False,
            device: torch.device = None,
            dtype: torch.dtype = None,
            *args,
            **kwargs
    ) -> None:
        kwargs.update(
            {
                "padding_idx": padding_idx,
                "max_norm": max_norm,
                "norm_type": norm_type,
                "scale_grad_by_freq": scale_grad_by_freq,
                "sparse": sparse,
                "_weight": _weight,
                "_freeze": _freeze,
                "device": device,
            }
        )

        super().__init__(
            in_features=num_embeddings,
            out_features=embeddings_dim,
            global_layers=global_layers,
            structure=structure,
            bias=False,
            dtype=dtype,
            *args,
            **kwargs
        )

    def _create_layer(
            self,
            **kwargs
    ) -> None:
        """
        Creates the layer of the class based on the structure attribute.
        The layer is created as product of global and local matrices, and it is at the end an Embedding layer.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            **kwargs:
                Arbitrary keyword arguments.
        """

        super()._create_layer(**kwargs)

    def _initialize_global_layers(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the global layers as Embedding or Linear layers.
        To approximate an Embedding layer, here the choice is to have an Embedding layer as first layer to extract the
        vector associated to an input index and then Linear layers to extract the output of the approximated Embedding
        layer.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            kwargs:
                Additional keyword arguments.
        """

        for layer_idx, layer in enumerate(self.structure):
            if layer["scope"] == "global":
                if layer["key"] not in self.global_layers.keys():
                    if layer_idx == 0:
                        self.global_layers[layer["key"]] = nn.Embedding(
                            *layer["shape"],
                            padding_idx=kwargs["padding_idx"],
                            max_norm=kwargs["max_norm"],
                            norm_type=kwargs["norm_type"],
                            scale_grad_by_freq=kwargs["scale_grad_by_freq"],
                            sparse=kwargs["sparse"],
                            _weight=kwargs["_weight"],
                            _freeze=kwargs["_freeze"],
                            device=kwargs["device"],
                            dtype=self.dtype
                        )
                    else:
                        self.global_layers[layer["key"]] = nn.Linear(
                            *layer["shape"],
                            bias=False,
                            device=kwargs["device"],
                            dtype=self.dtype
                        )

                    if "trainable" in layer.keys():
                        for param in self.global_layers[layer["key"]].parameters():
                            param.requires_grad = layer["trainable"]

    def _initialize_local_layers(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the global layers as Embedding or Linear layers.
        To approximate an Embedding layer, here the choice is to have an Embedding layer as first layer to extract the
        vector associated to an input index and then Linear layers to extract the output of the approximated Embedding
        layer.

        Args:
            bias (bool):
                Flag to include bias in the layer.
            kwargs:
                Additional keyword arguments.
        """

        self.local_layers = nn.ModuleDict()

        for layer_idx, layer in enumerate(self.structure):
            if layer["scope"] == "local":
                if layer["key"] not in self.local_layers.keys():
                    if layer_idx == 0:
                        self.local_layers[layer["key"]] = nn.Embedding(
                            *layer["shape"],
                            padding_idx=kwargs["padding_idx"],
                            max_norm=kwargs["max_norm"],
                            norm_type=kwargs["norm_type"],
                            scale_grad_by_freq=kwargs["scale_grad_by_freq"],
                            sparse=kwargs["sparse"],
                            _weight=kwargs["_weight"],
                            _freeze=kwargs["_freeze"],
                            device=kwargs["device"],
                            dtype=self.dtype
                        )
                    else:
                        self.local_layers[layer["key"]] = nn.Linear(
                            *layer["shape"],
                            bias=False,
                            device=kwargs["device"],
                            dtype=self.dtype
                        )

                    if "trainable" in layer.keys():
                        for param in self.local_layers[layer["key"]].parameters():
                            param.requires_grad = layer["trainable"]

    def merge(
            self,
            **kwargs
    ) -> nn.Embedding:
        """
        Merges the global and local layers into an equivalent Embedding layer.

        This method computes the equivalent Embedding layer by multiplying the weights of the global and local layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Equivalent linear Embedding with merged weights.

        Raises:
            Exception:
                If the layer's scope is neither 'global' nor 'local'.
        """

        equivalent_embedding = nn.Embedding(
            self.in_features,
            self.out_features,
        )

        weight = None

        with torch.no_grad():
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

            equivalent_embedding.weight.data = weight

        return equivalent_embedding

    @property
    def weight(
            self
    ) -> nn.Parameter:
        """
        Returns the weight parameter of the layer.

        Returns:
            Tensor:
                Weight parameter.

        Raises:
            Exception:
                If the layer's scope is neither 'global' nor 'local'.
        """

        if self.structure[0]["scope"] == "global":
            weight = self.global_layers[self.structure[0]["key"]].weight
        elif self.structure[0]["scope"] == "local":
            weight = self.local_layers[self.structure[0]["key"]].weight
        else:
            raise Exception("The last layer has to be global ('global') or local ('local').")

        for layer in self.structure[1:]:

            if layer["scope"] == "global":
                weight = torch.matmul(weight, self.global_layers[layer["key"]].weight.T)
            elif layer["scope"] == "local":
                weight = torch.matmul(weight, self.local_layers[layer["key"]].weight.T)
            else:
                raise Exception("The last layer has to be global ('global') or local ('local').")

        return weight

    @weight.setter
    def weight(
            self,
            value: torch.Tensor
    ) -> None:
        """
        Sets the weight parameter of the Embedding layer.

        For now, and probably forever, it is not implemented.

        Args:
            value (torch.Tensor):
                Weight parameter value.

        Raise:
            Exception:
                If the weight is tried to be set.
        """

        raise Exception('Cannot set the weight of a global dependent layer.')


class StructureSpecificGlobalDependentEmbedding(StructureSpecificGlobalDependent, ABC):
    """
    Abstract class that implements an Embedding layer with dependencies on global matrices that wraps a target Embedding
    layer.

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
        return GlobalDependentEmbedding(
            target_layer.num_embeddings,
            target_layer.embedding_dim,
            global_layers,
            structure,
            padding_idx=target_layer.padding_idx,
            max_norm=target_layer.max_norm,
            norm_type=target_layer.norm_type,
            scale_grad_by_freq=target_layer.scale_grad_by_freq,
            sparse=target_layer.sparse,
            device=target_layer.weight.device,
            dtype=target_layer.weight.dtype,
            **kwargs
        )

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


class LocalSVDEmbedding(StructureSpecificGlobalDependentEmbedding):
    """
    Implementation of an Embedding layer on which is performed SVD decomposition.

    The layer is decomposed in two matrices:
    - a local matrix of shape (num_embeddings, rank), that is the product between the matrix of the left singular
        vectors U and the matrix of the singular values S matrices of the truncated SVD, this will be the matrix of an
        Embedding layer;
    - a local matrix of shape (rank, embedding_dim), that is the matrix of the right singular vectors V^T, this will be
        the matrix of a Linear layer.

    Args:
        target_layer (nn.Module):
            Target layer to be transformed in the factorized version.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        rank (int):
            Rank of the factorization to use.
        target_name (str):
            Name of the target layer. Defaults to None.
        *args:
            Variable length argument list.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        target_name (str):
            Name of the target layer.
        global_dependent_layer (GlobalDependentLinear):
            Embedding layer with dependencies on global matrices.
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
                Additional keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]
        min_dim = min(target_layer.num_embeddings, target_layer.embedding_dim)

        return (
            {
                "scope": "local",
                "shape": (target_layer.num_embeddings, min(min_dim, rank)),
                "key": "US",
                "trainable": True
            },
            {
                "scope": "local",
                "shape": (min(min_dim, rank), target_layer.embedding_dim),
                "key": "VT",
                "trainable": True
            }
        )

    def initialize_matrices(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the matrices of the layer.

        Args:
            **kwargs:
                Additional keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        dtype = target_layer.weight.data.numpy().dtype
        rank = kwargs["rank"]

        U, S, VT = np.linalg.svd(target_layer.weight.data.numpy().astype(np.float32))
        min_dim = min(target_layer.num_embeddings, target_layer.embedding_dim)

        with torch.no_grad():
            self.get_layer("local", "US").weight.data = torch.tensor(
                U[:, :min(min_dim, rank)].astype(dtype) @ np.diag(S[:min(min_dim, rank)]).astype(dtype)
            )
            self.get_layer("local", "VT").weight.data = torch.tensor(
                VT[:min(min_dim, rank), :].astype(dtype)
            ).T

            for layer in self.structure:
                if "trainable" in layer.keys() and layer["scope"] == "local":
                    for params in self.get_layer("local", layer["key"]).parameters():
                        params.requires_grad = layer["trainable"]


class GlobalBaseEmbedding(StructureSpecificGlobalDependentEmbedding):
    """
    Implementation of an Embedding layer on which is performed matrix decomposition in two matrices:
    - a local matrix of shape (num_embeddings, rank), this will be the matrix of an Embedding layer;
    - a global trainable matrix of shape (rank, embedding_dim), this will be the matrix of a Linear layer.

    Args:
        target_layer (nn.Module):
            Target layer to be transformed in the factorized version.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        rank (int):
            Rank of the factorization to use.
        target_name (str):
            Name of the target layer. Defaults to None.
        *args:
            Variable length argument list.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        target_name (str):
            Name of the target layer.
        global_dependent_layer (GlobalDependentLinear):
            Embedding layer with dependencies on global matrices.
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
                Additional keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]

        return (
            {
                "scope": "local",
                "shape": (target_layer.num_embeddings, rank),
                "key": str((target_layer.num_embeddings, rank)),
                "trainable": True
            },
            {
                "scope": "global",
                "shape": (rank, target_layer.embedding_dim),
                "key": str((rank, target_layer.embedding_dim)),
                "trainable": True
            }
        )

    def initialize_matrices(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the matrices of the layer.

        Args:
            **kwargs:
                Additional keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        target_weight = target_layer.weight.data
        local_key = self.global_dependent_layer.structure[0]["key"]
        global_key = self.global_dependent_layer.structure[1]["key"]

        global_matrix = self.get_layer("global", global_key).weight.data.T

        with torch.no_grad():
            global_matrix = global_matrix.to(torch.float32)
            pinv_global_matrix = torch.pinverse(global_matrix.to("cpu"))
            #pinv_global_matrix = pinv_global_matrix.to(self.global_dependent_layer.dtype)

            local_matrix = target_weight.to(torch.float32) @ pinv_global_matrix
            local_matrix = local_matrix.to(self.global_dependent_layer.dtype)

            self.get_layer("local", local_key).weight.data = local_matrix

            if "trainable" in self.structure[0].keys():
                for params in self.get_layer("local", local_key).parameters():
                    params.requires_grad = self.structure[0]["trainable"]

        device = get_available_device()

        target_weight = target_weight.to(device)
        self.set_layer("global", global_key, self.get_layer("global", global_key).to(device))
        self.set_layer("local", local_key, self.get_layer("local", local_key).to(device))

        optimizer = torch.optim.AdamW(
            [
                self.get_layer("local", local_key).weight
            ],
            lr=1e-3,
            eps=1e-6 if target_weight.dtype == torch.float16 else 1e-8
        )

        num_epochs = 100

        for epoch in range(num_epochs):
            approximated_matrix = torch.matmul(
                self.get_layer("local", local_key).weight,
                self.get_layer("global", global_key).weight.T
            )

            loss = torch.norm((target_weight - approximated_matrix) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class GlobalFixedBaseEmbedding(GlobalBaseEmbedding):
    """
    Implementation of an Embedding layer on which is performed matrix decomposition in two matrices:
    - a local matrix of shape (num_embeddings, rank), this will be the matrix of an Embedding layer;
    - a global non-trainable matrix of shape (rank, embedding_dim), this will be the matrix of a Linear layer.

    The global matrix is fixed and not trainable.

    Args:
        target_layer (nn.Module):
            Target layer to be transformed in the factorized version.
        global_layers (nn.ModuleDict):
            Dictionary containing the global matrices.
        rank (int):
            Rank of the factorization to use.
        target_name (str, optional):
            Name of the target layer. Defaults to None.
        *args:
            Variable length argument list.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        target_name (str):
            Name of the target layer.
        global_dependent_layer (GlobalDependentLinear):
            Embedding layer with dependencies on global matrices.
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
        super().__init__(
            target_layer,
            global_layers,
            rank,
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
                Additional keyword arguments.
        """

        target_layer = kwargs["target_layer"]
        rank = kwargs["rank"]

        return (
            {
                "scope": "local",
                "shape": (target_layer.num_embeddings, rank),
                "key": str((target_layer.num_embeddings, rank)),
                "trainable": True
            },
            {
                "scope": "global",
                "shape": (rank, target_layer.embedding_dim),
                "key": str((rank, target_layer.embedding_dim)),
                "trainable": False
            }
        )


if __name__ == "__main__":

    embedding = nn.Embedding(
        num_embeddings=10000,
        embedding_dim=768,
        dtype=torch.float16
    )

    global_embedding = LocalSVDEmbedding(
        target_layer=embedding,
        global_layers=nn.ModuleDict(),
        average_layers=nn.ModuleDict(),
        target_name="embedding",
        rank=100,
    )

    print(embedding.weight)
    print(global_embedding.weight)
