import numpy as np

import torch.nn as nn


def compute_explained_variance(
    s: np.array, scaling: int = 1
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

    return (np.square(s) * scaling).cumsum() / (np.square(s) * scaling).sum()


def compute_rank(
        matrix: np.ndarray,
        threshold: float = 0,
        s_threshold: float = 0
) -> np.ndarray:
    """
    Computes the rank of a matrix.

    Args:
        matrix (np.ndarray):
            The matrix to compute the rank of.
        threshold (float):
            The threshold to use to compute the rank.

    Returns:
        np.ndarray:
            The rank of the matrix.
    """

    _, s, _ = np.linalg.svd(matrix)
    explained_variance = compute_explained_variance(s)
    rank_based_on_explained_variance = np.argmax(explained_variance > threshold)
    if s[-1] < s_threshold:
        rank_based_on_explained_variance = len(explained_variance)

    rank_based_on_singular_values = np.argmax(s < s_threshold)
    if s[-1] > s_threshold:
        rank_based_on_singular_values = len(s)

    rank = np.minimum(rank_based_on_explained_variance, rank_based_on_singular_values)

    return rank


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
                        "weight": child.weight,
                        "layer_name": layer_name,
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


def compute_max_possible_rank(
        matrices: list
) -> int:
    """
    Computes the maximum possible rank for a list of delta matrices.

    Args:
        matrices (list):
            The list of delta matrices.

    Returns:
        int:
            The maximum possible rank.
    """

    max_possible_rank = 0
    for matrix in matrices:
        shape = matrix["weight"].shape
        max_dim = max(shape)
        max_possible_rank = max(max_possible_rank, max_dim)

    return max_possible_rank
