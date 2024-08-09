import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from gbm.utils.printing_utils.printing_utils import Verbose


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
