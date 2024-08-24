from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import transformers
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import torch
import torch.nn as nn

import re

from neuroflex.utils.printing_utils.printing_utils import Verbose


# Definition of the classes to perform the rank analysis

class RankAnalysisResult:
    """
    Class to store the result of the rank analysis. It stores the rank of the tensor and the thresholds used to compute
    the rank.

    Args:
        rank (int):
            The rank of the tensor.
        explained_variance_threshold (float, optional):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance. Defaults to 0.
        singular_values_threshold (float, optional):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold. Defaults to 0.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.

    Attributes:
        rank (int):
            The rank of the tensor.
        explained_variance_threshold (float):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance.
        singular_values_threshold (float):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold.
        verbose (Verbose):
            The verbosity level.
    """

    def __init__(
            self,
            rank: int,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0,
            verbose: Verbose = Verbose.INFO
    ) -> None:

        self.rank = rank
        self.explained_variance_threshold = explained_variance_threshold
        self.singular_values_threshold = singular_values_threshold

        self.verbose = verbose

    def get_rank(
            self
    ) -> int:
        """
        Returns the rank of the tensor.

        Returns:
            int:
                The rank of the tensor.
        """

        return self.rank

    def get_explained_variance_threshold(
            self
    ) -> float:
        """
        Returns the threshold on the explained variance to use to compute the rank.

        Returns:
            float:
                The threshold on the explained variance to use to compute the rank.
        """

        return self.explained_variance_threshold

    def get_singular_values_threshold(
            self
    ) -> float:
        """
        Returns the threshold to use to compute the rank based on singular values.

        Returns:
            float:
                The threshold to use to compute the rank based on singular values.
        """

        return self.singular_values_threshold


class AnalysisTensorWrapper:
    """
    Wrapper for the analysis of a tensor.

    Args:
        tensor (torch.Tensor):
            The tensor.
        name (str, optional):
            The name of the tensor. Defaults to None.
        label (str, optional):
            The label of the tensor. Defaults to None.
        path (str, optional):
            The path of the tensor. Defaults to None.
        block_index (int, optional):
            The block index of the tensor. Defaults to None.
        layer (nn.Module, optional):
            The layer of the tensor. Defaults to None.
        precision (int, optional):
            The precision of the relative rank of the tensor. Defaults to 2.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.

    Attributes:
        tensor (torch.Tensor):
            The tensor.
        name (str):
            The name of the tensor.
        label (str):
            The label of the tensor.
        path (str):
            The path of the tensor.
        block_index (int):
            The block index of the tensor.
        layer (nn.Module):
            The layer of the tensor.
        singular_values (np.ndarray):
            The singular values of the tensor.
        rank_analysis_results (list):
            The list of rank analysis
    """

    def __init__(
            self,
            tensor: torch.Tensor,
            name: str = None,
            label: str = None,
            path: str = None,
            block_index: int = None,
            layer: nn.Module = None,
            precision: int = 2,
            verbose: Verbose = Verbose.INFO
    ) -> None:

        self.tensor = tensor
        self.name = name
        self.label = label
        self.path = path
        self.block_index = block_index
        self.layer = layer

        self.precision = precision

        self.verbose = verbose

        self.singular_values = None
        self.rank_analysis_results = []

        self.attributes = {}

    def get_tensor(
            self,
            numpy_array: bool = False
    ) -> [np.ndarray | torch.Tensor]:
        """
        Returns the tensor.

        Returns:
            [np.ndarray | torch.Tensor]:
                The tensor.
        """

        if numpy_array:
            return self.tensor.detach().numpy()
        else:
            return self.tensor.detach()

    def get_name(
            self
    ) -> str:
        """
        Returns the name of the tensor.

        Returns:
            str:
                The name of the tensor.
        """

        return self.name

    def get_label(
            self
    ) -> str:
        """
        Returns the label of the tensor.

        Returns:
            str:
                The label of the tensor.
        """

        return self.label

    def get_path(
            self
    ) -> str:
        """
        Returns the path of the tensor.

        Returns:
            str:
                The path of the tensor.
        """

        return self.path

    def get_block_index(
            self
    ) -> int:
        """
        Returns the block index of the tensor.

        Returns:
            int:
                The block index of the tensor.
        """

        return self.block_index

    def get_layer(
            self
    ) -> nn.Module:
        """
        Returns the layer of the tensor.

        Returns:
            nn.Module:
                The layer of the tensor.
        """

        return self.layer

    def get_precision(
            self
    ) -> int:
        """
        Returns the precision of the relative rank of the tensor.

        Returns:
            int:
                The precision of the relative rank of the tensor.
        """

        return self.precision

    def get_verbose(
            self
    ) -> Verbose:
        """
        Returns the verbosity level.

        Returns:
            Verbose:
                The verbosity level.
        """

        return self.verbose

    def get_singular_values(
            self
    ) -> np.ndarray:
        """
        Returns the singular values of the tensor.

        Returns:
            np.ndarray:
                The singular values of the tensor.
        """

        return self.singular_values

    def get_explained_variance(
            self
    ) -> None:
        """
        Returns the explained variance of the tensor.

        Returns:
            np.ndarray:
                The explained variance of the tensor.
        """

        return compute_explained_variance(self.singular_values)

    def get_rank(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0,
            relative: bool = True
    ) -> [int | float]:
        """
        Returns the rank of the tensor.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance. Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold. Defaults to 0.
            relative (bool, optional):
                Whether to return the relative rank. Defaults to True.

        Returns:
            [int | float]:
                The rank of the tensor.
        """

        for rank_analysis_result in self.rank_analysis_results:
            if rank_analysis_result.get_explained_variance_threshold() == explained_variance_threshold and \
                    rank_analysis_result.get_singular_values_threshold() == singular_values_threshold:
                rank = rank_analysis_result.get_rank()
                if relative:
                    shape = self.get_shape()
                    rank = round(rank / (torch.sqrt(torch.tensor(shape[0]) * torch.tensor(shape[1]))).item(), self.precision)
                return rank

        self._perform_rank_analysis(explained_variance_threshold, singular_values_threshold)

        rank = self.get_rank(explained_variance_threshold, singular_values_threshold, relative)

        return rank

    def get_shape(
            self
    ) -> torch.Size:
        """
        Returns the shape of the tensor.

        Returns:
            torch.Size:
                The shape of the tensor.
        """

        return self.tensor.shape

    def get_attribute(
            self,
            key: str
    ) -> Any:
        """
        Returns the attribute given the key.

        Args:
            key (str):
                The key of the attribute.

        Returns:
            Any:
                The attribute.
        """

        return self.attributes[key]

    def get_parameters_count(
            self
    ) -> int:
        """
        Returns the number of parameters of the tensor.

        Returns:
            int:
                The number of parameters of the tensor.
        """

        return self.tensor.numel()

    def get_parameters_count_thresholded(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0
    ) -> int:
        """
        Returns the number of parameters of the tensor.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance. Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold. Defaults to 0.

        Returns:
            int:
                The number of parameters of the tensor.
        """

        rank = self.get_rank(explained_variance_threshold, singular_values_threshold, relative=False)

        return rank * self.get_shape()[0] + rank * self.get_shape()[1]

    def set_tensor(
            self,
            tensor: torch.Tensor
    ) -> None:
        """
        Sets the tensor.

        Args:
            tensor (torch.Tensor):
                The tensor.
        """

        self.tensor = tensor

    def set_dtype(
            self,
            dtype: torch.dtype
    ) -> None:
        """
        Sets the dtype of the tensor.

        Args:
            dtype (torch.dtype):
                The dtype of the tensor.
        """

        self.tensor = self.tensor.to(dtype)

    def set_device(
            self,
            device: torch.device
    ) -> None:
        """
        Sets the device of the tensor.

        Args:
            device (torch.device):
                The device of the tensor.
        """

        self.tensor = self.tensor.to(device)

    def set_name(
            self,
            name: str
    ) -> None:
        """
        Sets the name of the tensor.

        Args:
            name (str):
                The name of the tensor.
        """

        self.name = name

    def set_label(
            self,
            label: str
    ) -> None:
        """
        Sets the label of the tensor.

        Args:
            label (str):
                The label of the tensor.
        """

        self.label = label

    def set_path(
            self,
            path: str
    ) -> None:
        """
        Sets the path of the tensor.

        Args:
            path (str):
                The path of the tensor.
        """

        self.path = path

    def set_block_index(
            self,
            block_index: int
    ) -> None:
        """
        Sets the block index of the tensor.

        Args:
            block_index (int):
                The block index of the tensor.
        """

        self.block_index = block_index

    def set_layer(
            self,
            layer: nn.Module
    ) -> None:
        """
        Sets the layer of the tensor.

        Args:
            layer (nn.Module):
                The layer of the tensor.
        """

        self.layer = layer

    def set_singular_values(
            self,
            singular_values: np.ndarray
    ) -> None:
        """
        Sets the singular values of the tensor.

        Args:
            singular_values (np.ndarray):
                The singular values of the tensor.
        """

        self.singular_values = singular_values

    def set_attribute(
            self,
            key: str,
            value: Any
    ) -> None:
        """
        Sets the attribute given the key.

        Args:
            key (str):
                The key of the attribute.
            value (Any):
                The value of the attribute.
        """

        self.attributes[key] = value

    def append_rank_analysis_result(
            self,
            rank_analysis_result: RankAnalysisResult
    ) -> None:
        """
        Appends a rank analysis result.

        Args:
            rank_analysis_result (RankAnalysisResult):
                The rank analysis result.
        """

        self.rank_analysis_results.append(rank_analysis_result)

    def delete_rank_analysis_result(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0
    ) -> None:
        """
        Deletes a rank analysis result.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance. Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold. Defaults to 0.
        """

        for rank_analysis_result in self.rank_analysis_results:
            if rank_analysis_result.get_explained_variance_threshold() == explained_variance_threshold and \
                    rank_analysis_result.get_singular_values_threshold() == singular_values_threshold:
                self.rank_analysis_results.remove(rank_analysis_result)

    def delete_rank_analyses(
            self
    ) -> None:
        """
        Deletes all rank analyses.
        """

        self.rank_analysis_results = []

    def compute_singular_values(
            self
    ) -> None:
        """
        Computes the singular values of the tensor.
        """

        self.set_singular_values(compute_singular_values(self.get_tensor(numpy_array=True)))

    def _perform_rank_analysis(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0
    ) -> None:
        """
        Performs the rank analysis of the tensor.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance.
                Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold.
                Defaults to 0.
        """

        if self.singular_values is None:
            raise ValueError("Singular values not computed.")

        if explained_variance_threshold <= 0. or explained_variance_threshold > 1.:
            raise ValueError("The threshold on the explained variance must be between 0 and 1.")

        explained_variance = compute_explained_variance(self.singular_values)

        rank_based_on_explained_variance = np.argmax(explained_variance >= explained_variance_threshold) + 1

        if self.singular_values[-1] > singular_values_threshold:
            rank_based_on_singular_values = len(self.singular_values)
        else:
            rank_based_on_singular_values = np.argmax(self.singular_values < singular_values_threshold)

        rank = np.minimum(
            rank_based_on_explained_variance,
            rank_based_on_singular_values
        )

        self.append_rank_analysis_result(
            RankAnalysisResult(
                rank=rank,
                explained_variance_threshold=explained_variance_threshold,
                singular_values_threshold=singular_values_threshold
            )
        )


class AnalysisTensorDict:
    """
    Dictionary of tensors for the analysis.

    Args:
        keys ([list[tuple[Any, ...]] | list[Any]], optional):
            The keys of the tensors. Defaults to None.
        tensors ([list[list[AnalysisTensorWrapper]] | list[AnalysisTensorWrapper]], optional):
            The tensors to add to the dictionary.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.

    Raises:
        ValueError:
            If the number of keys is different from the number of tensors.
        ValueError:
            If all keys do not have the same length.

    Attributes:
        tensors (dict):
            The dictionary of tensors.
        verbose (Verbose):
            The verbosity level.
    """

    def __init__(
            self,
            keys: [list[tuple[Any, ...]] | list[Any]] = (),
            tensors: [list[list[AnalysisTensorWrapper]] | list[AnalysisTensorWrapper]] = (),
            verbose: Verbose = Verbose.INFO
    ) -> None:

        if len(tensors) != len(keys):
            raise ValueError("The number of keys must be equal to the number of tensors.")
        if len(keys) > 0:
            for index in range(len(keys)):
                if not isinstance(keys[index], tuple):
                    keys[index] = (keys[index],)
            keys_length = len(keys[0])
            for key in keys:
                if len(key) != keys_length:
                    raise ValueError("All keys must have the same length.")

        self.tensors = {}
        for index in range(len(keys)):
            self.append_tensor(
                keys[index],
                tensors[index]
            )

        self.verbose = verbose

    def get_tensor(
            self,
            key: [tuple[Any, ...] | Any],
            index: int = 0
    ) -> AnalysisTensorWrapper:
        """
        Returns the tensor given the key.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensor.
            index (int, optional):
                The index of the tensor in the list. Defaults to 0.

        Returns:
            AnalysisTensorWrapper:
                The tensor.
        """

        if not isinstance(key, tuple):
            key = (key,)

        return self.tensors[key][index]

    def get_tensor_list(
            self,
            key: [tuple[Any, ...] | Any]
    ) -> list[AnalysisTensorWrapper]:
        """
        Returns the list of tensors given the key.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensors.

        Returns:
            list[AnalysisTensorWrapper]:
                The list of tensors.
        """

        if not isinstance(key, tuple):
            key = (key,)

        return self.tensors[key]

    def get_keys(
            self
    ) -> list[tuple[Any, ...]]:
        """
        Returns the keys of the dictionary.

        Returns:
            list[tuple[Any, ...]]:
                The keys of the dictionary.
        """

        return list(self.tensors.keys())

    def get_unique_positional_keys(
            self,
            position: int,
            sort: bool = False
    ) -> list[Any]:
        """
        Returns the unique keys at a given position.

        Args:
            position (int):
                The position of the keys.
            sort (bool, optional):
                Whether to return the keys sorted. Defaults to False.

        Returns:
            list[Any]:
                The unique keys at the given position.
        """

        unique_keys = list(set(
            key[position]
            for key in self.tensors.keys()
        ))

        if sort:
            unique_keys.sort()

        return unique_keys

    def set_tensor(
            self,
            key: [tuple[Any, ...] | Any],
            tensor: [list[AnalysisTensorWrapper] | AnalysisTensorWrapper]
    ) -> None:
        """
        Sets a tensor or a list of tensors to the dictionary.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensor.
            tensor ([list[AnalysisTensorWrapper] | AnalysisTensorWrapper]):
                The tensor or list of tensors to set.

        Raises:
            ValueError:
                If the key is not a tuple.
        """

        if not isinstance(key, tuple):
            key = (key,)

        if isinstance(tensor, list):
            self.tensors[key] = tensor
        else:
            self.tensors[key] = [tensor]

    def append_tensor(
            self,
            key: [tuple[Any, ...] | Any],
            tensor: [list[AnalysisTensorWrapper] | AnalysisTensorWrapper]
    ) -> None:
        """
        Appends a tensor to the dictionary.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensor.
            tensor (AnalysisTensorWrapper):
                The tensor or list of tensors to append.
        """

        if not isinstance(key, tuple):
            key = (key,)

        if key in self.tensors.keys():
            if isinstance(tensor, list):
                self.tensors[key] = self.tensors[key] + tensor
            else:
                self.tensors[key] = self.tensors[key] + [tensor]
        else:
            self.set_tensor(key, tensor)

    def set_dtype(
            self,
            dtype: torch.dtype
    ) -> None:
        """
        Sets the dtype of the tensors.

        Args:
            dtype (torch.dtype):
                The dtype of the tensors.
        """

        for key in self.tensors.keys():
            for tensor_wrapper in self.tensors[key]:
                tensor_wrapper.set_dtype(dtype)

    def set_device(
            self,
            device: torch.device
    ) -> None:
        """
        Sets the device of the tensors.

        Args:
            device (torch.device):
                The device of the tensors.
        """

        for key in self.tensors.keys():
            for tensor_wrapper in self.tensors[key]:
                tensor_wrapper.set_device(device)

    def filter_by_positional_key(
            self,
            key: Any,
            position: int
    ) -> 'AnalysisTensorDict':
        """
        Filters the tensors given a key and a position.

        Args:
            key (Any):
                The key to filter.
            position (int):
                The position of the key.

        Returns:
            AnalysisTensorDict:
                The filtered dictionary of tensors.
        """

        filtered_tensors = AnalysisTensorDict()
        for tensor_key in self.tensors.keys():
            if tensor_key[position] == key:
                filtered_tensors.append_tensor(
                    tensor_key,
                    self.tensors[tensor_key]
                )

        return filtered_tensors


class AnalysisLayerWrapper(nn.Module):
    """
    Wrapper for the analysis of a layer.

    Args:
        layer (nn.Module):
            The layer.
        label (str, optional):
            The label of the layer. Defaults to None.

    Attributes:
        layer (nn.Module):
            The layer.
        label (str):
            The label of the layer.
        activations (list):
            The list of activations.
        mean_activations (torch.Tensor):
            The mean of the activations.
        variance_activations (torch.Tensor):
            The variance of the activations.
    """

    def __init__(
            self,
            layer: nn.Module,
            label: str = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layer = layer
        self.label = label

        self.activations = []
        self.mean_activations = None
        self.variance_activations = None

    def get_layer(
            self
    ) -> nn.Module:
        """
        Returns the layer.

        Returns:
            nn.Module:
                The layer.
        """

        return self.layer

    def get_label(
            self
    ) -> str:
        """
        Returns the label of the layer.

        Returns:
            str:
                The label of the layer.
        """

        return self.label

    def get_activations(
            self
    ) -> list:
        """
        Returns the activations.

        Returns:
            list:
                The activations.
        """

        return self.activations

    def get_mean_activations(
            self
    ) -> torch.Tensor:
        """
        Returns the mean of the activations.

        Returns:
            torch.Tensor:
                The mean of the activations.
        """

        return self.mean_activations

    def get_variance_activations(
            self
    ) -> torch.Tensor:
        """
        Returns the variance of the activations.

        Returns:
            torch.Tensor:
                The variance of the activations.
        """

        return self.variance_activations

    def set_label(
            self,
            label: str
    ) -> None:
        """
        Sets the label of the layer.

        Args:
            label (str):
                The label of the layer.
        """

        self.label = label

    def set_activations(
            self,
            activations: list
    ) -> None:
        """
        Sets the activations.

        Args:
            activations (list):
                The activations.
        """

        self.activations = activations

    def set_mean_activations(
            self,
            mean_activations: torch.Tensor
    ) -> None:
        """
        Sets the mean of the activations.

        Args:
            mean_activations (torch.Tensor):
                The mean of the activations.
        """

        self.mean_activations = mean_activations

    def set_variance_activations(
            self,
            variance_activations: torch.Tensor
    ) -> None:
        """
        Sets the variance of the activations.

        Args:
            variance_activations (torch.Tensor):
                The variance of the activations.
        """

        self.variance_activations = variance_activations

    def compute_mean_activations(
            self
    ) -> None:
        """
        Computes the mean of the activations.
        """

        self.set_mean_activations(torch.mean(torch.stack(self.activations), dim=0))

    def compute_variance_activations(
            self
    ) -> None:
        """
        Computes the variance of the activations.
        """

        self.set_variance_activations(torch.var(torch.stack(self.activations), dim=0))

    def compute_stats(
            self
    ) -> None:
        """
        Computes the statistics of the activations.
        """

        self.compute_mean_activations()
        self.compute_variance_activations()

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor):
                The input tensor.

        Returns:
            torch.Tensor:
                The output tensor.
        """

        output = self.layer(x, *args, **kwargs)
        self.activations.append(output)

        return output


class AnalysisModelWrapper(nn.Module):
    """
    Wrapper for the analysis of a model.

    Args:
        model (nn.Module):
            The model.
        *args:
            Additional positional arguments.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        model (nn.Module):
            The model.
    """

    def __init__(
            self,
            model: [nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.wrap_model(self.model)

    def wrap_model(
            self,
            module_tree: nn.Module,
            path: str = "",
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        """
        Converts layers into global-dependent versions.

        Args:
            module_tree (nn.Module):
                Model or module containing layers.
            path (str):
                Path to the current layer.
            verbose (Verbose):
                Level of verbosity.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            # If the child has no children, the layer is an actual computational layer of the model
            if len(child._modules) == 0:
                # Creating a wrapper for the layer
                layer_wrapper = AnalysisLayerWrapper(child, path + (f"{layer_name}" if path == "" else f"_{layer_name}"))
                # Setting the wrapper as the child of the current module
                module_tree._modules[layer_name] = layer_wrapper
            else:
                # Recursively calling the method on the child, if the child has children
                self._convert_into_global_dependent_model(
                    child,
                    path + (f"{layer_name}" if path == "" else f"_{layer_name}"),
                    verbose=verbose,
                    **kwargs
                )

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor):
                The input tensor.

        Returns:
            torch.Tensor:
                The output tensor.
        """

        return self.model(x, *args, **kwargs)


# Definition of the mathematical functions to perform the rank analysis

def compute_singular_values(
        matrix: np.ndarray
) -> np.ndarray:
    """
    Computes the singular values of a matrix.

    Args:
        matrix (np.ndarray):
            The matrix to compute the singular values of.

    Returns:
        np.ndarray:
            The singular values of the matrix.
    """

    return np.linalg.svd(matrix, compute_uv=False)


def compute_explained_variance(
    s: np.array,
    scaling: int = 1
) -> np.array:
    """
    Computes the explained variance for a set of singular values.

    Args:
        s (np.array):
            The singular values.
        scaling (float, optional):
            Scaling to apply to the explained variance at each singular value.
            Defaults to 1.

    Returns:
        np.array:
            The explained variance for each singular value.
    """

    if s[0] == 0.:
        return np.ones(len(s))

    return (np.square(s) * scaling).cumsum() / (np.square(s) * scaling).sum()


def compute_rank(
        singular_values: dict,
        threshold: float = 0,
        s_threshold: float = 0
) -> dict:
    """
    Computes the rank of a matrix considering negligible eigenvalues that are very small or that provide a very small
    change in terms of fraction of explained variance.

    Args:
        singular_values (dict):
            The singular values of the matrices of the model.
            The dictionary has the following structure:
            >> {
            >>    "layer_name": {
            >>        "s": [np.array, ...]
            >>    }
            >> }
        threshold (float):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance.
        s_threshold (float):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold.

    Returns:
        dict:
            The ranks given the input singular values.
    """

    ranks = {}
    for layer_name in singular_values.keys():
        ranks[layer_name] = []

        for s in singular_values[layer_name]["s"]:
            explained_variance = compute_explained_variance(s)
            rank_based_on_explained_variance = np.argmax(explained_variance > threshold)
            if s[-1] < s_threshold:
                rank_based_on_explained_variance = len(explained_variance)

            rank_based_on_singular_values = np.argmax(s < s_threshold)
            if s[-1] > s_threshold:
                rank_based_on_singular_values = len(s)

            rank = np.minimum(rank_based_on_explained_variance, rank_based_on_singular_values)

            ranks[layer_name].append(rank)

    return ranks


def compute_max_possible_rank(
        analyzed_matrices: AnalysisTensorDict
) -> int:
    """
    Computes the maximum possible rank of the matrices of the model.

    Args:
        analyzed_matrices (AnalysisTensorDict):
            The analyzed matrices of the model.

    Returns:
        int:
            The maximum possible rank of the matrices of the model.
    """

    max_possible_rank = 0
    for key in analyzed_matrices.get_keys():
        for analyzed_matrix in analyzed_matrices.get_tensor_list(key):
            singular_values = analyzed_matrix.get_singular_values()
            max_possible_rank = max(max_possible_rank, len(singular_values))

    return max_possible_rank


# Definition of the functions to extract the matrices from the model tree

def extract(
        model_tree: nn.Module,
        names_of_targets: list,
        extracted_matrices: list,
        path: list = [],
        verbose: bool = False,
        **kwargs
) -> None:
    """
    Extracts the matrices from the model tree.

    Args:
        model_tree (nn.Module):
            The model tree.
        names_of_targets (list):
            The names of the targets.
        extracted_matrices (list):
            The list of extracted matrices.
        path (list, optional):
            The path to the current layer. Defaults to [].
        verbose (bool, optional):
            Whether to print the layer name. Defaults to False.
    """

    for layer_name in model_tree._modules.keys():
        child = model_tree._modules[layer_name]
        if len(child._modules) == 0:
            if layer_name in names_of_targets:
                if verbose:
                    print(f"Found {layer_name} in {path}")

                extracted_matrices.append(
                    {
                        "weight": child.weight.detach().numpy(),
                        "layer_name": layer_name,
                        "label": [el for el in path if re.search(r'\d', el)][0],
                        "path": path
                    }
                )
        else:
            new_path = path.copy()
            new_path.append(layer_name)
            extract(
                child,
                names_of_targets,
                extracted_matrices,
                new_path,
                verbose=verbose,
                **kwargs
            )


def extract_based_on_path(
        model_tree: [nn.Module | transformers.AutoModel],
        paths_of_targets: list,
        extracted_matrices: list,
        black_list: list = None,
        path: str = "",
        verbose: Verbose = Verbose.INFO,
        **kwargs
) -> None:
    """
    Extracts the matrices from the model tree.

    Args:
        model_tree ([nn.Module | transformers.AutoModel]):
            The model tree.
        paths_of_targets (list):
            The path of the targets.
        extracted_matrices (list):
            The list of extracted matrices.
        black_list (list, optional):
            The list of black listed paths. Defaults to None.
        path (str, optional):
            The path to the current layer. Defaults to "".
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.
    """

    for layer_name in model_tree._modules.keys():
        child = model_tree._modules[layer_name]
        if len(child._modules) == 0:
            if verbose > Verbose.INFO:
                print(f"Checking {layer_name} in {path}")

            if black_list is not None:
                black_listed = len([
                    black_listed_string
                    for black_listed_string in black_list
                    if black_listed_string in path + "_" + layer_name
                ]) > 0
            else:
                black_listed = False

            targets_in_path = [
                layer_path_
                for layer_path_ in paths_of_targets
                if layer_path_ in path + "_" + layer_name and not black_listed
            ]
            if len(targets_in_path) > 0:
                layer_path = str(max(targets_in_path, key=len))
                if verbose > Verbose.SILENT:
                    print(f"Found {layer_path} in {path}")

                list_containing_layer_number = [
                    sub_path for sub_path in path.split("_") if sub_path.isdigit()
                ]
                block_index = list_containing_layer_number[0] if len(list_containing_layer_number) > 0 else "-1"
                extracted_matrices.append(
                    AnalysisTensorWrapper(
                        tensor=child.weight.detach(),
                        name=layer_name,
                        label=layer_path,
                        path=path,
                        block_index=int(block_index),
                        layer=child
                    )
                )
        else:
            # Recursively calling the function
            extract_based_on_path(
                model_tree=child,
                paths_of_targets=paths_of_targets,
                extracted_matrices=extracted_matrices,
                black_list=black_list,
                path=path + "_" + layer_name,
                verbose=verbose,
                **kwargs
            )


# Definition of the functions to plot the results of the rank analysis

def plot_heatmap(
        data: np.ndarray,
        interval: dict,
        title: str = "Rank analysis of the matrices of the model",
        x_title: str = "Layer indexes",
        y_title: str = "Type of matrix",
        columns_labels: list = None,
        rows_labels: list = None,
        figure_size: tuple = (24, 5),
        save_path: str = None,
        heatmap_name: str = "heatmap",
        show: bool = False
):
    """
    Plots a heatmap.

    Args:
        data (np.ndarray):
            The data to plot.
        interval (dict):
            The interval to use to plot the heatmap.
        title (str, optional):
            The title of the heatmap. Defaults to "Rank analysis of the matrices of the model".
        x_title (str, optional):
            The title of the x-axis. Defaults to "Layer indexes".
        y_title (str, optional):
            The title of the y-axis. Defaults to "Type of matrix".
        columns_labels (list, optional):
            The labels of the columns. Defaults to None.
        rows_labels (list, optional):
            The labels of the rows. Defaults to None.
        figure_size (tuple, optional):
            The size of the figure. Defaults to (10, 5).
        save_path (str, optional):
            The path where to save the heatmap. Defaults to None.
        heatmap_name (str, optional):
            The name of the heatmap. Defaults to "heatmap".
        show (bool, optional):
            Whether to show the heatmap. Defaults to False.
    """

    # Creating the figure
    fig, axs = plt.subplots(
        1,
        1,
        figsize=figure_size
    )

    # Showing the heatmap
    heatmap = axs.imshow(
        data,
        cmap="hot",
        interpolation="nearest",
        vmin=interval["min"],
        vmax=interval["max"]
    )

    # Setting title, labels and ticks
    axs.set_title(
        title,
        fontsize=18,
        y=1.05
    )
    axs.set_xlabel(
        x_title,
        fontsize=12
    )
    axs.set_ylabel(
        y_title,
        fontsize=12
    )
    if rows_labels:
        axs.set_yticks(np.arange(len(rows_labels)))
        axs.set_yticklabels(rows_labels)
    if columns_labels:
        axs.set_xticks(np.arange(len(columns_labels)))
        axs.set_xticklabels(columns_labels)
    axs.axis("on")

    # Adding the colorbar
    divider = make_axes_locatable(axs)
    colormap_axis = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        heatmap,
        cax=colormap_axis
    )

    # Changing the color of the text based on the background to make it more readable
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > interval["min"] + (interval["max"] - interval["min"]) / 3:
                text_color = "black"
            else:
                text_color = "white"
            axs.text(j, i, f"{data[i, j]}", ha="center", va="center", color=text_color)

    plt.tight_layout()

    # Storing the heatmap
    if save_path and os.path.exists(save_path):
        plt.savefig(
            os.path.join(
                save_path,
                heatmap_name
            )
        )
        print("Heatmap stored to", os.path.join(save_path, heatmap_name))

    # Showing the heatmap
    if show:
        plt.show()

    plt.close()


# Definition of the functions to manage and check the storage

def check_path_to_storage(
        path_to_storage: str,
        type_of_analysis: str,
        model_name: str,
        strings_to_be_in_the_name: tuple
) -> tuple[bool, str, str]:
    """
    Checks if the path to the storage exists.
    If the path exists, the method returns a positive flag and the path to the storage of the experiment data.
    If the path does not exist, the method returns a negative flag and creates the path for the experiment returning it.

    Args:
        path_to_storage (str):
            The path to the storage where the experiments data have been stored or will be stored.
        type_of_analysis (str):
            The type of analysis to be performed on the model.
        model_name (str):
            The name of the model to analyze.
        strings_to_be_in_the_name (tuple):
            The strings to be used to create the name or to find in the name of the stored data of the considered
            experiment.

    Returns:
        bool:
            A flag indicating if the path to the storage of the specific experiment already exists.
        str:
            The path to the storage of the specific experiment.
        str:
            The name of the file to store the data.

    Raises:
        Exception:
            If the path to the storage does not exist.
    """

    if not os.path.exists(path_to_storage):
        raise Exception(f"The path to the storage '{path_to_storage}' does not exist.")

    # Checking if the path to the storage of the specific experiment already exists
    exists_directory_path = os.path.exists(
        os.path.join(
            path_to_storage, model_name
        )
    ) & os.path.isdir(
        os.path.join(
            path_to_storage, model_name
        )
    ) & os.path.exists(
        os.path.join(
            path_to_storage, model_name, type_of_analysis
        )
    ) & os.path.isdir(
        os.path.join(
            path_to_storage, model_name, type_of_analysis
        )
    )

    exists_file = False
    directory_path = os.path.join(
        path_to_storage, model_name, type_of_analysis
    )
    file_name = None
    if exists_directory_path:
        try:
            files_and_dirs = os.listdir(
                directory_path
            )

            # Extracting the files
            files = [
                f
                for f in files_and_dirs
                if os.path.isfile(os.path.join(path_to_storage, model_name, type_of_analysis, f))
            ]

            # Checking if some file ame contains the required strings
            for f_name in files:
                names_contained = all(string in f_name for string in strings_to_be_in_the_name)
                if names_contained:
                    exists_file = True
                    file_name = f_name
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
            return False, "", ""

    else:
        os.makedirs(
            os.path.join(
                directory_path
            )
        )

    if not exists_file:
        file_name = "_".join(strings_to_be_in_the_name)

    return exists_file, directory_path, str(file_name)