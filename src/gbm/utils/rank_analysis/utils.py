import os
from time import sleep

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
            The title of the x axis. Defaults to "Layer indexes".
        y_title (str, optional):
            The title of the y axis. Defaults to "Type of matrix".
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
