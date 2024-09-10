from __future__ import annotations

import os
import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import torch

from neuroflex.utils.printing_utils.printing_utils import Verbose


def get_text_color(
        value: float,
        cmap: mcolors.Colormap
) -> str:
    """
    Get the color of the text based on the value.

    Args:
        value (float):
            The value to determine the color.
        cmap (mcolors.Colormap):
            The colormap to use to determine the color.
    """

    rgba = cmap(value)
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    return "white" if luminance < 0.5 else "black"


def get_label_from_list(
        label: str | list,
        index: int
) -> str:
    """
    Returns the label from a list if it is a list. Otherwise, returns the label.

    Args:
        label (str | list):
            The label or list of labels.
        index (int):
            The index of the label to return.

    Returns:
        str:
            The label.
    """

    return label[index] if isinstance(type(label), list) else label


def set_axis_labels(
        ax: plt.Axes,
        x_title: str = "Column Index",
        y_title: str = "Row Index",
        x_labels: list = None,
        y_labels: list = None,
        x_ticks: list = None,
        y_ticks: list = None,
        x_ticks_visible: bool = True,
        y_ticks_visible: bool = True,
        x_rotation: float = 0,
        y_rotation: float = 0
) -> None:
    """
    Sets the labels of the axes of a plot.

    Args:
        ax (plt.Axes):
            The axes of the plot.
        x_title (str):
            The title of the x-axis. Default: "Column Index".
        y_title (str):
            The title of the y-axis. Default: "Row Index".
        x_labels (list):
            The labels of the x-axis. Default: None.
        y_labels (list):
            The labels of the y-axis. Default: None.
        x_ticks (list):
            The ticks of the x-axis. Default: None.
        y_ticks (list):
            The ticks of the y-axis. Default: None.
        x_ticks_visible (bool):
            Whether to display the ticks on the x-axis. Default: True.
        y_ticks_visible (bool):
            Whether to display the ticks on the y-axis. Default: True.
        x_rotation (float):
            The rotation of the x-axis labels. Default: 0.
        y_rotation (float):
            The rotation of the y-axis labels. Default: 0.
    """

    # Setting the titles of the axes
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    # Setting the ticks and labels of the axes
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)) if x_ticks is None else x_ticks)
        ax.set_xticklabels(x_labels, rotation=x_rotation)
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)) if y_ticks is None else y_ticks)
        ax.set_yticklabels(y_labels, rotation=y_rotation)

    # Removing ticks from the borders
    if not x_ticks_visible:
        ax.xaxis.set_ticks_position("none")
    if not y_ticks_visible:
        ax.yaxis.set_ticks_position("none")


def plot_heatmap(
        value_matrices_lists: list[list[np.ndarray | torch.Tensor]],
        save_path: str,
        title: str,
        axis_titles: list[str] = None,
        x_title: str | list[str] = "Column Index",
        y_title: str | list[str] = "Row Index",
        x_labels: list[list[str]] = None,
        y_labels: list[list[str]] = None,
        cmap_str: str = "Blues",
        fig_size: tuple = (20, 20),
        precision: int = 2
) -> None:
    """
    Plot a heatmap without additional row and column.

    Args:
        value_matrices_lists (list[list[np.ndarray]]):
            The list of matrices to plot.
        save_path (str):
            The path where to save the plot.
        title (str):
            The title of the plot.
        axis_titles (list[str]):
            The titles of the x-axis and y-axis. Default: None.
        x_title (str):
            The title of the x-axis. Default: "Column Index".
        y_title (str):
            The title of the y-axis. Default: "Row Index".
        x_labels (list[list[str]]):
            The labels of the x-axis. Default: None.
        y_labels (list[list[str]]):
            The labels of the y-axis. Default: None.
        fig_size (tuple):
            The size of the figure. Default: (20, 20).
        cmap_str (str):
            The colormap to use. Default: "Blues".
        precision (int):
            The number of decimal places to display in cell values. Default: 2.
    """

    if len(value_matrices_lists) <= 0:
        raise ValueError("At least one matrix must be provided.")
    if not all([len(value_matrices_list) > 0 for value_matrices_list in value_matrices_lists]):
        raise ValueError("All lists of matrices must contain at least one matrix.")

    if not all([value_matrix.shape == value_matrices_list[0].shape
                for value_matrices_list in value_matrices_lists
                for value_matrix in value_matrices_list]):
        raise ValueError("All matrices in the same list (to be plotted together) must have the same shape.")

    # Getting the number of axis to be created
    num_axis = len(value_matrices_lists)

    # Creating the plot
    fig, axs = plt.subplots(1, num_axis, figsize=fig_size)
    fig.suptitle(title)

    for axis_index in range(num_axis):
        if num_axis == 1:
            ax = axs
        else:
            ax = axs[axis_index]
        if axis_titles is not None:
            ax.set_title(axis_titles[axis_index])

        value_matrices_list = value_matrices_lists[axis_index]

        # Getting the number of rows and columns
        num_rows, num_cols = value_matrices_list[0].shape

        # Displaying the matrix
        cax = ax.matshow(value_matrices_list[0], cmap=cmap_str,
                         norm=mcolors.Normalize(vmin=value_matrices_list[0].min(), vmax=value_matrices_list[0].max()))

        # Adding color bar
        fig.colorbar(cax)

        # Adding labels
        set_axis_labels(ax, get_label_from_list(x_title, axis_index), get_label_from_list(y_title, axis_index),
                        x_labels[axis_index] if x_labels else None, y_labels[axis_index] if y_labels else None,
                        x_ticks=list(range(num_cols)), y_ticks=list(range(num_rows)), x_ticks_visible=False, y_ticks_visible=False, x_rotation=90)

        # Adding text annotations for each cell
        for i in range(num_rows):
            for j in range(num_cols):
                cell_string = "\n".join([f"{value_matrix[i, j]:.{precision}f}" for value_matrix in value_matrices_list])
                ax.text(j, i, f"{cell_string}", ha="center", va="center", color=get_text_color(float(value_matrices_list[0][i, j]), plt.get_cmap(cmap_str)))

    # Adjusting layout and show plot
    plt.tight_layout()

    # Saving the plot
    plt.savefig(save_path)
    logging.info(f"Heatmap saved at '{save_path}'")


def plot_heatmap_with_additional_row_column(
        value_matrices_lists: list[list[np.ndarray | torch.Tensor]],
        values_rows_lists: list[list[np.ndarray | torch.Tensor]],
        values_columns_lists: list[list[np.ndarray | torch.Tensor]],
        save_path: str,
        title: str,
        axis_titles: list[str] = None,
        x_title: str | list[str] = "Column Index",
        y_title: str | list[str] = "Row Index",
        x_labels: list[list[str]] = None,
        y_labels: list[list[str]] = None,
        cmap_str: str = "Blues",
        fig_size: tuple = (20, 20),
        precision: int = 2
) -> None:
    """
    Plot a heatmap with an additional row and column to display additional information.

    Args:
        value_matrices_lists (list[list[np.ndarray | torch.Tensor]]):
            The list of matrices to plot.
        values_rows_lists (list[list[np.ndarray | torch.Tensor]]):
            The list of values to display in the first row.
        values_columns_lists (list[list[np.ndarray | torch.Tensor]]):
            The list of values to display in the first column.
        save_path (str):
            The path where to save the plot.
        title (str):
            The title of the plot.
        axis_titles (list[str]):
            The titles of the x-axis and y-axis. Default: None.
        x_title (str):
            The title of the x-axis. Default: "Column Index".
        y_title (str):
            The title of the y-axis. Default: "Row Index".
        x_labels (list[list[str]]):
            The labels of the x-axis. Default: None.
        y_labels (list[list[str]]):
            The labels of the y-axis. Default: None.
        cmap_str (str):
            The colormap to use. Default: "Blues".
        fig_size (tuple):
            The size of the figure. Default: (14, 14).
    """

    if len(value_matrices_lists) <= 0:
        raise ValueError("At least one matrix must be provided.")
    if len(value_matrices_lists) != len(values_rows_lists) or len(value_matrices_lists) != len(values_columns_lists):
        raise ValueError("The number of matrices, rows values, and columns values must be the same.")
    if not all([len(value_matrices_list) > 0 for value_matrices_list in value_matrices_lists]):
        raise ValueError("All lists of matrices must contain at least one matrix.")

    if not all([value_matrix.shape == value_matrices_list[0].shape
                for value_matrices_list in value_matrices_lists
                for value_matrix in value_matrices_list]):
        raise ValueError("All matrices in the same list (to be plotted together) must have the same shape.")

    # Getting the number of axis to be created
    num_axis = len(value_matrices_lists)

    # Creating the plot
    fig, ax = plt.subplots(1, num_axis, figsize=fig_size)
    fig.suptitle(title)

    for axis_index in range(num_axis):
        if axis_titles is not None:
            ax[axis_index].set_title(axis_titles[axis_index])

        value_matrices_list, values_rows_list, values_columns_list = value_matrices_lists[axis_index], values_rows_lists[axis_index], values_columns_lists[axis_index]

        # Getting the number of rows and columns
        num_rows, num_cols = value_matrices_list[0].shape

        extended_matrices = []
        for value_matrix, values_row, values_column in zip(value_matrices_list, values_rows_list, values_columns_list):
            # Creating an extended matrices with an extra row and column for indices
            extended_matrix = np.zeros((num_rows + 1, num_cols + 1))
            extended_matrix[1:, 1:] = value_matrix
            extended_matrix[0, 1:] = values_row
            extended_matrix[1:, 0] = values_column
            extended_matrices.append(extended_matrix)

        # Masking the first cell to be invisible
        mask = np.zeros_like(extended_matrices[0], dtype=bool)
        mask[0, 0] = True
        mask[1:, 0] = False
        mask[0, 1:] = False

        # Displaying the matrix with a masked array
        cax = ax[axis_index].matshow(np.ma.masked_array(extended_matrices[0], mask=mask), cmap=cmap_str,
                         norm=mcolors.Normalize(vmin=extended_matrices[0][1:, 1:].min(), vmax=extended_matrices[0][1:, 1:].max()))

        # Adding color bar
        fig.colorbar(cax)

        # Adding labels
        set_axis_labels(ax[axis_index], get_label_from_list(x_title, axis_index), get_label_from_list(y_title, axis_index), x_labels[axis_index], y_labels[axis_index], x_ticks=np.arange(num_cols + 1).tolist(), y_ticks=np.arange(num_rows + 1).tolist(), x_ticks_visible=False, y_ticks_visible=False, x_rotation=90)

        # Hiding the border lines of the matrix
        ax[axis_index].spines["top"].set_visible(False)
        ax[axis_index].spines["left"].set_visible(False)

        # Adding thicker border between the first row and the others, starting from the second column
        # Horizontal line below the first row, starting from column 1
        ax[axis_index].axhline(y=0.5, color="black", linewidth=4, xmin=1 / (num_cols + 1), xmax=1)
        # Vertical line to the right of the first column
        ax[axis_index].axvline(x=0.5, color="black", linewidth=4, ymin=0, ymax=num_rows / (num_rows + 1))

        # Adding text annotations for each cell (excluding the first cell)
        for i in range(num_rows + 1):
            for j in range(num_cols + 1):
                if not mask[i, j]:
                    cell_string = "\n".join([f"{extended_matrix[i, j]:.2f}" for extended_matrix in extended_matrices])
                    ax[axis_index].text(j, i, f"{cell_string}", ha="center", va="center", color=get_text_color(float(extended_matrices[0][i, j]), plt.get_cmap(cmap_str)))

    # Adjusting layout and show plot
    plt.tight_layout()

    # Saving the plot
    plt.savefig(save_path)
    logging.info(f"Heatmap saved at '{save_path}'")


def create_heatmap_global_layers(
        data: list,
        title: str = "Global layers used in different parts of the model",
        x_title: str = "Layer indexes",
        y_title: str = "Type of matrix",
        columns_labels: list = None,
        rows_labels: list = None,
        figure_size: tuple = (25, 15),
        save_path: str = None,
        heatmap_name: str = "heatmap",
        label_to_index: dict = None,
        verbose: Verbose = Verbose.SILENT,
        show: bool = False
) -> dict:
    """
    Plots a heatmap using a different color for each layer that uses a different global matrix.

    Args:
        data (list of lists):
            The data to plot. Each element is a list of categorical labels.
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
        label_to_index (dict, optional):
            A dictionary that maps the labels to numerical values. Defaults to None.
        show (bool, optional):
            Whether to show the heatmap. Defaults to False.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.SILENT.

    Returns:
        dict:
            A dictionary that maps the labels to numerical values.
    """

    if label_to_index is None:
        # Flattening the data to get unique labels
        flat_data = [item for sublist in data for item in sublist]
        unique_labels = list(set(flat_data))

        # Creating a numerical mapping for the labels
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Converting the data to numerical data
    numerical_data = np.array([[label_to_index[label] for label in row] for row in data])

    # Determine the min and max values for the colormap
    vmin = min(label_to_index.values())
    vmax = max(label_to_index.values())

    # Create a colormap
    # Generate a custom colormap with enough distinct colors
    # Generate colors using a gradient or another method
    base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    num_colors = len(label_to_index)
    if num_colors <= len(base_colors):
        colors = base_colors[:num_colors]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

    color_map = mcolors.ListedColormap(colors)

    # Create the figure
    fig, axs = plt.subplots(
        1,
        1,
        figsize=figure_size
    )

    # Show the heatmap
    heatmap = axs.imshow(
        numerical_data,
        cmap=color_map,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )

    # Set title, labels, and ticks
    axs.set_title(
        title,
        fontsize=20,
        y=1.05
    )
    axs.set_xlabel(
        x_title,
        fontsize=15
    )
    axs.set_ylabel(
        y_title,
        fontsize=15
    )
    if rows_labels:
        axs.set_yticks(np.arange(len(rows_labels)))
        axs.set_yticklabels(rows_labels, fontsize=13)
    if columns_labels:
        axs.set_xticks(np.arange(len(columns_labels)))
        axs.set_xticklabels(columns_labels, fontsize=13)
    axs.axis("on")

    # Adding the colorbar
    divider = make_axes_locatable(axs)
    colormap_axis = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        heatmap,
        cax=colormap_axis,
        ticks=np.arange(vmin, vmax + 1),
        format='%d'
    )
    plt.tight_layout()

    # Storing the heatmap
    if save_path and os.path.exists(save_path):
        plt.savefig(
            os.path.join(
                save_path,
                heatmap_name
            )
        )
        if verbose > Verbose.INFO:
            print("Heatmap stored to", os.path.join(save_path, heatmap_name))

    # Showing the heatmap
    if show:
        plt.show()

    plt.close()

    return label_to_index
