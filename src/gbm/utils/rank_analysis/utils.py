import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import re

import torch.nn as nn


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
        singular_values: dict
) -> int:
    """
    Computes the maximum possible rank for a list of delta matrices.

    Args:
        singular_values (dict):
            The singular values of the matrices of the model.

    Returns:
        int:
            The maximum possible rank.
    """

    max_possible_rank = 0
    for layer_name in singular_values.keys():
        for singular_values_one_matrix in singular_values[layer_name]["s"]:
            max_possible_rank = max(max_possible_rank, len(singular_values_one_matrix))

    return max_possible_rank


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


def plot_heatmap(
        data: np.ndarray,
        interval: dict,
        figure_title: str = "Heatmaps",
        titles: list = (
                "Map with colour bar between maximum and maximum rank value",
                "Map with colore bar between 0 and the maximum rank value"
        ),
        x_title: str = "Layer indexes",
        y_title: str = "Type of matrix",
        columns_labels: list = None,
        rows_labels: list = None,
        figure_size: tuple = (10, 8)
):
    """
    Plots a heatmap.

    Args:
        data (np.ndarray):
            The data to plot.
        interval (dict):
            The interval in which input data can assume values.
        figure_title (str, optional):
            The title of the figure. Defaults to "Heatmaps".
        titles (list, optional):
            The titles of the two heatmaps. Defaults to (
                "Map with colour bar between maximum and maximum rank value",
                "Map with colore bar between 0 and the maximum rank value"
            ).
        x_title (str, optional):
            The title of the x axis. Defaults to "Layer indexes".
        y_title (str, optional):
            The title of the y axis. Defaults to "Type of matrix".
        columns_labels (list, optional):
            The labels of the columns. Defaults to None.
        rows_labels (list, optional):
            The labels of the rows. Defaults to None.
        figure_size (tuple, optional):
            The size of the figure. Defaults to (10, 8).
    """

    fig, axs = plt.subplots(2, 1, figsize=figure_size)
    fig.suptitle(figure_title)

    im0 = axs[0].imshow(data, cmap="hot", interpolation="nearest")
    axs[0].set_title(titles[0])
    axs[0].set_xlabel(x_title)
    axs[0].set_ylabel(y_title)
    if rows_labels:
        axs[0].set_yticks(np.arange(len(rows_labels)))
        axs[0].set_yticklabels(rows_labels)
    if columns_labels:
        axs[0].set_xticks(np.arange(len(columns_labels)))
        axs[0].set_xticklabels(columns_labels)
    axs[0].set_xticks(np.arange(0, data.shape[1]))
    axs[0].tick_params(axis="x", which='both', length=0)
    axs[0].tick_params(axis="y", which="both", length=0)
    axs[0].axis("on")
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > np.min(data) + (np.max(data) - np.min(data)) / 2:
                text_color = "black"
            else:
                text_color = "white"
            axs[0].text(j, i, f"{data[i, j]}", ha="center", va="center", color=text_color)

    im1 = axs[1].imshow(data, cmap="hot", interpolation="nearest", vmin=interval["min"], vmax=interval["max"])
    axs[1].set_title(titles[1])
    axs[1].set_xlabel(x_title)
    axs[1].set_ylabel(y_title)
    if rows_labels:
        axs[1].set_yticks(np.arange(len(rows_labels)))
        axs[1].set_yticklabels(rows_labels)
    if columns_labels:
        axs[1].set_xticks(np.arange(len(columns_labels)))
        axs[1].set_xticklabels(columns_labels)
    axs[1].tick_params(axis="x", which="both", length=0)
    axs[1].tick_params(axis="y", which="both", length=0)
    axs[1].axis("on")
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > interval["min"] + (interval["max"] - interval["min"]) / 2:
                text_color = "black"
            else:
                text_color = "white"
            axs[1].text(j, i, f"{data[i, j]}", ha="center", va="center", color=text_color)

    plt.tight_layout()
    plt.show()

