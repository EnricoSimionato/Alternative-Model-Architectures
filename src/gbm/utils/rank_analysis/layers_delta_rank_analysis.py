import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import torch.nn as nn

from transformers import AutoModelForCausalLM

from gbm.utils.rank_analysis.utils import compute_rank, extract, compute_max_possible_rank


def compute_delta_consecutive_matrices(
        matrices: list
) -> list:
    """
    Compute the delta between consecutive matrices.

    Args:
        matrices (list):
            List of matrices.

    Returns:
        list:
            List of delta matrices.
    """

    delta_matrices = []
    for i in range(len(matrices) - 1):
        matrix_1 = matrices[i]["weight"]
        matrix_2 = matrices[i + 1]["weight"]

        delta_matrix = matrix_2 - matrix_1
        delta_matrices.append(
            {
                "delta_matrix": delta_matrix,
                "layers": str(i + 1) + " - " + str(i)
            }
        )

    return delta_matrices


def plot_heatmap(
        data: np.ndarray,
        interval: dict,
        titles: list = (
                "Map with colour bar between maximum and maximum rank value",
                "Map with colore bar between 0 and the maximum rank value"
        ),
        x_title: str = "Layers numbers of delta",
        y_title: str = "Type of matrix",
        columns_labels: list = None,
        rows_labels: list = None,
        figure_size: tuple = (10, 8)
):
    fig, axs = plt.subplots(2, 1, figsize=figure_size)

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


def start_layers_delta_rank_analysis(
        model: nn.Module,
        layers_to_analyze=(
            "query",
            "key",
            "value"
        ),
        threshold: float = 0.9,
        s_threshold: float = 0,
        figure_size: tuple = (10, 8),
        verbose: bool = False
):

    ranks = []
    for layer_name in layers_to_analyze:
        extracted_matrices = []
        extract(
            model,
            [layer_name],
            extracted_matrices,
        )

        delta_matrices = compute_delta_consecutive_matrices(
            extracted_matrices
        )

        for delta_matrix in delta_matrices:
            rank = compute_rank(
                delta_matrix["delta_matrix"].detach().numpy(),
                threshold,
                s_threshold
            )
            ranks.append(rank)
            if verbose:
                print(f"Rank for {layer_name} {delta_matrix['layers']}: {rank}")

    plot_heatmap(
        np.array(ranks).reshape(len(layers_to_analyze), -1),
        interval={"min": 0, "max": compute_max_possible_rank(extracted_matrices)},
        rows_labels=layers_to_analyze,
        columns_labels=[el["layers"] for el in delta_matrices],
        figure_size=figure_size
    )


if __name__ == "__main__":
    #import huggingface_hub
    #huggingface_hub.login("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")
    """
    model_to_analyse = AutoModelForCausalLM.from_pretrained("gpt2")
    print(model_to_analyse)
    start_layers_delta_rank_analysis(
        model_to_analyse,
        layers_to_analyze=(
            "query",
            "key",
            "value"
        ),
        threshold=0.9,
        #s_threshold=0.1
    )
    """
    a = np.array(
        [[100,200,300,400,500,600,200,240,333,111,231,342,200,240,333,111,231,342,200,240,333,111,231,342,111,231,342,200,240,333,111,231,342],
         [200,240,333,111,231,342,200,240,333,111,231,342,200,240,333,111,231,342,200,240,333,111,231,342,111,231,342,200,240,333,111,231,342],
         [200,240,333,111,231,342,200,240,333,111,231,342,200,240,333,111,231,342,200,240,333,111,231,342,111,231,342,200,240,333,111,231,342]]
    )
    plot_heatmap(
        a,
        interval={"min": 0, "max": 800},
        rows_labels=["a", "b"],
        figure_size=(16, 8)
    )



