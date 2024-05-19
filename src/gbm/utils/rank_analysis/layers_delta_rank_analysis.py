import numpy as np

import torch.nn as nn

from transformers import AutoModelForCausalLM

from gbm.utils.rank_analysis.utils import (
    compute_rank,
    extract,
    compute_max_possible_rank,

    plot_heatmap
)


def compute_delta_matrices(
        minuend_matrices: list,
        subtrahend_matrices: list
) -> list:
    """
    Computes the delta between two lists of matrices.

    Args:
        minuend_matrices (list):
            List of minuend matrices.
        subtrahend_matrices (list):
            List of subtrahend matrices.

    Returns:
        list:
            List of delta matrices.
    """

    delta_matrices = []
    for i in range(len(minuend_matrices)):
        minuend_matrix = minuend_matrices[i]["weight"]
        subtrahend_matrix = subtrahend_matrices[i]["weight"]

        delta_matrix = minuend_matrix - subtrahend_matrix
        delta_matrices.append(
            {
                "delta_matrix": delta_matrix,
                "delta_label": str(minuend_matrices[i]["label"]) + " - " + str(subtrahend_matrices[i]["label"])
            }
        )

    return delta_matrices


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

    minuend_matrices = matrices[1:].copy()
    subtrahend_matrices = matrices[:-1].copy()

    return compute_delta_matrices(
        minuend_matrices,
        subtrahend_matrices
    )


def compute_delta_average_matrices(
        matrices: list
) -> list:
    """
    Compute the delta between the average matrix and the rest of the matrices.

    Args:
        matrices (list):
            List of matrices.

    Returns:
        list:
            List of delta matrices.
    """

    minuend_matrices = matrices.copy()
    average_matrix = np.mean([el["weight"] for el in matrices], axis=0)
    layer_name = f"{matrices[0]['layer_name']}"
    for i in range(len(matrices)):
        layer_name += "_"
        layer_name += matrices[i]["layer_name"]

    average_matrix = {
        "weight": average_matrix,
        "layer_name": layer_name,
        "label": "avg",
        "path": []
    }

    subtrahend_matrices = [average_matrix] * len(minuend_matrices)

    return compute_delta_matrices(
        minuend_matrices,
        subtrahend_matrices
    )


def start_layers_rank_analysis(
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
    """
    Does the rank analysis of the layers and plots the heatmap.

    Args:
        model (nn.Module):
            The model.
        layers_to_analyze (tuple):
            The layers to analyze.
        threshold (float):
            The threshold.
        s_threshold (float):
            The threshold for the singular values.
        figure_size (tuple):
            The figure size.
        verbose (bool):
            Whether to print the rank.
    """

    ranks = []
    for layer_name in layers_to_analyze:
        extracted_matrices = []
        extract(
            model,
            [layer_name],
            extracted_matrices,
        )

        for extracted_matrix in extracted_matrices:
            rank = compute_rank(
                extracted_matrix["weight"],
                threshold,
                s_threshold
            )
            ranks.append(rank)
            if verbose:
                print(f"Rank for {layer_name} {extracted_matrix['label']}: {rank}")

    plot_heatmap(
        np.array(ranks).reshape(len(layers_to_analyze), -1),
        interval={"min": 0, "max": compute_max_possible_rank(extracted_matrices)},
        figure_title="Ranks of matrices of the layers",
        rows_labels=layers_to_analyze,
        columns_labels=[el["label"] for el in extracted_matrices],
        figure_size=figure_size
    )


def start_layers_delta_average_rank_analysis(
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
        """
        Does the rank analysis of the layers and plots the heatmap.

        Args:
            model (nn.Module):
                The model.
            layers_to_analyze (tuple):
                The layers to analyze.
            threshold (float):
                The threshold.
            s_threshold (float):
                The threshold for the singular values.
            figure_size (tuple):
                The figure size.
            verbose (bool):
                Whether to print the rank.
        """

        ranks = []
        for layer_name in layers_to_analyze:
            extracted_matrices = []
            extract(
                model,
                [layer_name],
                extracted_matrices,
            )

            delta_matrices = compute_delta_average_matrices(
                extracted_matrices
            )

            for delta_matrix in delta_matrices:
                rank = compute_rank(
                    delta_matrix["delta_matrix"],
                    threshold,
                    s_threshold
                )
                ranks.append(rank)
                if verbose:
                    print(f"Rank for {layer_name} {delta_matrix['delta_label']}: {rank}")

        plot_heatmap(
            np.array(ranks).reshape(len(layers_to_analyze), -1),
            interval={"min": 0, "max": compute_max_possible_rank(extracted_matrices)},
            figure_title="Ranks of delta matrices with respect to the average matrix",
            rows_labels=layers_to_analyze,
            columns_labels=[el["delta_label"] for el in delta_matrices],
            figure_size=figure_size
        )


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
    """
    Does the rank analysis of the layers and plots the heatmap.

    Args:
        model (nn.Module):
            The model.
        layers_to_analyze (tuple):
            The layers to analyze.
        threshold (float):
            The threshold.
        s_threshold (float):
            The threshold for the singular values.
        figure_size (tuple):
            The figure size.
        verbose (bool):
            Whether to print the rank.
    """

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
                delta_matrix["delta_matrix"],
                threshold,
                s_threshold
            )
            ranks.append(rank)
            if verbose:
                print(f"Rank for {layer_name} {delta_matrix['delta_label']}: {rank}")

    plot_heatmap(
        np.array(ranks).reshape(len(layers_to_analyze), -1),
        interval={"min": 0, "max": compute_max_possible_rank(extracted_matrices)},
        figure_title="Ranks of delta matrices with respect to the matrix in the previous layer",
        rows_labels=layers_to_analyze,
        columns_labels=[el["delta_label"] for el in delta_matrices],
        figure_size=figure_size
    )


if __name__ == "__main__":
    model_to_analyse = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
    print(model_to_analyse)
    start_layers_rank_analysis(
        model_to_analyse,
        layers_to_analyze=(
            "query",
            "key",
            "value"
        ),
        threshold=0.9,
        #s_threshold=0.1
    )
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

    start_layers_delta_average_rank_analysis(
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
    """

