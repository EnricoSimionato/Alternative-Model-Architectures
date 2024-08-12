import os
import pickle
import sys

import numpy as np
import torch

import torch.nn as nn

from transformers import AutoModelForCausalLM

from gbm.utils.printing_utils.printing_utils import Verbose

from gbm.utils.experiment_pipeline import Config

from gbm.utils.rank_analysis.utils import (
    AnalysisTensorWrapper,
    AnalysisTensorDict
)

from gbm.utils.rank_analysis.utils import (
    compute_rank,
    extract,
    compute_max_possible_rank,

    plot_heatmap,
    compute_singular_values,
    check_path_to_storage
)


# Definition of the functions to compute the delta between layers

def compute_delta_matrices(
        minuend_matrices: list[AnalysisTensorWrapper],
        subtrahend_matrices: list[AnalysisTensorWrapper],
        verbose: Verbose = Verbose.SILENT
) -> list[AnalysisTensorWrapper]:
    """
    Computes the delta between two lists of matrices.

    Args:
        minuend_matrices (list):
            List of minuend matrices.
        subtrahend_matrices (list):
            List of subtrahend matrices.
        verbose (Verbose):
            The verbosity level. Defaults to Verbose.SILENT.

    Returns:
        list[AnalysisTensorWrapper]:
            List of delta matrices.
    """

    if verbose >= Verbose.DEBUG:
        print("Computing delta matrices...")

    delta_matrices = []

    for i in range(len(minuend_matrices)):
        minuend_matrix = minuend_matrices[i].get_tensor()
        subtrahend_matrix = subtrahend_matrices[i].get_tensor()

        delta_matrix = minuend_matrix - subtrahend_matrix
        delta_matrices.append(
            AnalysisTensorWrapper(
                delta_matrix,
                name=minuend_matrices[i].get_name(),
                label=str(minuend_matrices[i].get_label()) + " - " + str(subtrahend_matrices[i].get_label()),
                block_index=minuend_matrices[i].get_block_index()
            )
        )

    return delta_matrices


def compute_delta_consecutive_matrices(
        matrices: list[AnalysisTensorWrapper],
        verbose: Verbose = Verbose.SILENT
) -> list[AnalysisTensorWrapper]:
    """
    Compute the delta between consecutive matrices.

    Args:
        matrices (list[AnalysisTensorWrapper]):
            List of matrices.
        verbose (Verbose):
            The verbosity level. Defaults to Verbose.SILENT.

    Returns:
        list:
            List of delta matrices.
    """

    if verbose >= Verbose.DEBUG:
        print("Computing delta consecutive matrices...")

    minuend_matrices = matrices[1:].copy()
    subtrahend_matrices = matrices[:-1].copy()

    return compute_delta_matrices(
        minuend_matrices,
        subtrahend_matrices
    )


def compute_delta_wrt_average_matrices(
        matrices: list[AnalysisTensorWrapper],
        verbose: Verbose = Verbose.SILENT
) -> list[AnalysisTensorWrapper]:
    """
    Compute the delta between the average matrix and the rest of the matrices.

    Args:
        matrices (list[AnalysisTensorWrapper]):
            List of matrices.
        verbose (Verbose):
            The verbosity level. Defaults to Verbose.SILENT.

    Returns:
        list[AnalysisTensorWrapper]:
            List of delta matrices.
    """

    if verbose >= Verbose.DEBUG:
        print("Computing delta matrices with respect to the average matrix...")

    minuend_matrices = matrices.copy()
    stacked_tensors = torch.stack([matrix.get_tensor() for matrix in matrices])
    average_tensor = torch.mean(stacked_tensors, dim=0)
    layer_name = f"{matrices[0].get_name()}"

    for i in range(len(matrices)):
        layer_name += "_"
        layer_name += matrices[i].get_name()

    average_matrix = AnalysisTensorWrapper(
        average_tensor,
        name=layer_name,
        label="avg",
        block_index=matrices[0].get_block_index(),
        path=matrices[0].get_path()
    )

    subtrahend_matrices = [average_matrix] * len(minuend_matrices)

    return compute_delta_matrices(
        minuend_matrices,
        subtrahend_matrices
    )


# Definition of the functions to start the layers analysis of the delta matrices between consecutive layers

def start_layers_delta_rank_analysis(
        model: nn.Module,
        model_name: str,
        layers_to_analyze: tuple,
        path_to_storage: str,
        verbose: Verbose = Verbose.SILENT
):
    """
    Computes and stores the singular values of the deltas between consecutive layers.

    Args:
        model (nn.Module):
            The model to analyze.
        model_name (str):
            The name of the model.
        layers_to_analyze (tuple):
            The layers to analyze.
        path_to_storage (str):
            The path where to store the singular values of the layers.
        verbose (bool):
            Whether to print some additional information.
    """

    if not os.path.exists(path_to_storage):
        raise Exception(f"The path '{path_to_storage}' does not exist.")

    s_delta_layers = {}
    for layer_name in layers_to_analyze:
        extracted_matrices = []
        extract(
            model,
            [layer_name],
            extracted_matrices
        )

        delta_matrices = compute_delta_consecutive_matrices(
            extracted_matrices
        )

        s_delta_layers[layer_name] = {
            "s": [],
            "labels": []
        }

        for delta_matrix in delta_matrices:
            s = compute_singular_values(
                delta_matrix["matrix"]
            )
            s_delta_layers[layer_name]["s"].append(s)
            s_delta_layers[layer_name]["labels"].append(delta_matrix["label"])
            if verbose >= Verbose.DEBUG:
                print("Singular values for", layer_name, delta_matrix['label'], "extracted")

    if not os.path.exists(os.path.join(path_to_storage, model_name)):
        os.makedirs(os.path.join(path_to_storage, model_name))
    with open(
            os.path.join(path_to_storage, model_name, "_".join(["delta_consecutive_layers", "_".join(layers_to_analyze)])),
            'wb'
    ) as f:
        pickle.dump(s_delta_layers, f)

    return s_delta_layers


def perform_delta_consecutive_layers_rank_analysis(
        configuration: Config,
        verbose: Verbose = Verbose.SILENT
):
    """
    Performs the rank analysis of the difference between consecutive layers of the model.

    Args:
        configuration:
            The configuration object containing the necessary information.
        verbose:
            The verbosity level. Default is Verbose.SILENT.
    """

    pass

# Definition of the functions to start the layers analysis of the delta matrices between layers and the average matrix

def start_layers_delta_average_rank_analysis(
        model: nn.Module,
        model_name: str,
        layers_to_analyze: tuple,
        path_to_storage: str,
        verbose: Verbose = Verbose.SILENT
):
    """
    Computes and stores the singular values of the deltas between layers and the average matrix of the layers.

    Args:
        model (nn.Module):
            The model to analyze.
        model_name (str):
            The name of the model.
        layers_to_analyze (tuple):
            The layers to analyze.
        path_to_storage (str):
            The path where to store the singular values of the layers.
        verbose (Verbose):
            The verbosity level. Defaults to Verbose.SILENT.
    """

    if not os.path.exists(path_to_storage):
        raise Exception(f"The path '{path_to_storage}' does not exist.")

    s_delta_layers_wrt_average = {}
    for layer_name in layers_to_analyze:
        extracted_matrices = []
        extract(
            model,
            [layer_name],
            extracted_matrices
        )

        delta_matrices = compute_delta_wrt_average_matrices(
            extracted_matrices
        )

        s_delta_layers_wrt_average[layer_name] = {
            "s": [],
            "labels": []
        }

        for delta_matrix in delta_matrices:
            s = compute_singular_values(
                delta_matrix["matrix"]
            )
            s_delta_layers_wrt_average[layer_name]["s"].append(s)
            s_delta_layers_wrt_average[layer_name]["labels"].append(delta_matrix["label"])
            if verbose >= Verbose.DEBUG:
                print("Singular values for", layer_name, delta_matrix['label'], "extracted")

    if not os.path.exists(os.path.join(path_to_storage, model_name)):
        os.makedirs(os.path.join(path_to_storage, model_name))
    with open(
            os.path.join(
                path_to_storage,
                model_name,
                "_".join(["delta_layers_wrt_average", "_".join(layers_to_analyze)])
            ),
            'wb'
    ) as f:
        pickle.dump(s_delta_layers_wrt_average, f)

    return s_delta_layers_wrt_average


def perform_delta_layers_wrt_average_rank_analysis(
        configuration: Config,
        verbose: Verbose = Verbose.SILENT
):
    """
    Performs the rank analysis of the difference between matrices of the model and the average matrix.

    Args:
        configuration:
            The configuration object containing the necessary information.
        verbose:
            The verbosity level. Default is Verbose.SILENT.
    """

    pass


def start_layers_analysis(
        type_of_analysis: str,
        model_id: str,
        model: nn.Module = None,
        layers_to_analyze: tuple = None,
        threshold: float = 0.9,
        s_threshold: float = 0,
        relative_plot: bool = True,
        figure_size: tuple = (24, 5),
        path_to_storage: str = None,
        verbose: Verbose = Verbose.SILENT,
        heatmap_save_path: str = None,
):
    """
    Starts the layers analysis.

    Args:
        type_of_analysis (str):
            The type of analysis.
        model_id (str):
            The model id.
        model (nn.Module, optional):
            The model to analyze. Defaults to None.
        layers_to_analyze (tuple, optional):
            The layers to analyze. Defaults to None.
        threshold (float, optional):
            The threshold. Defaults to 0.9.
        s_threshold (float, optional):
            The threshold for the singular values. Defaults to 0.
        figure_size (tuple, optional):
            The size of the figure. Defaults to (10, 8).
        path_to_storage (str, optional):
            The path to the storage. Defaults to None.
        verbose (Verbose):
            The verbosity level. Defaults to Verbose.SILENT.
        heatmap_save_path (str, optional):
            The path where to save the heatmap. Defaults to None.

    Raises:
        Exception:
            If the type of analysis is not supported.

    Notes:
        The format of the s_dict dictionary is the following:
        {
            layer_name: {
                "s": list,
                "labels": list
            }
        }
        The format of the ranks dictionary is the following:
        {
            layer_name: list
        }

    """

    model_name = model_id.split("/")[-1]

    file_available, file_path = check_path_to_storage(
        path_to_storage,
        model_name,
        type_of_analysis,
        layers_to_analyze
    )
    s_dict = {}

    if not file_available:
        print("No data storage to use for the analysis found.")
        print("Performing svd and then running the analysis.")

        if model is None:
            model_to_analyse = AutoModelForCausalLM.from_pretrained(model_id)
        else:
            model_to_analyse = model

    else:
        print("Data storage to use for the analysis found.")

        # Load the data from the storage
        with open(file_path, 'rb') as f:
            s_dict = pickle.load(f)
            print(f"Data loaded from '{file_path}'")

    if type_of_analysis == "original_layers":
        title = f"Ranks of matrices of the layers (expl. var.: {threshold}, min. sing. value: {s_threshold})"

        if not file_available:
            s_dict = start_original_layers_rank_analysis(
                model_to_analyse,
                model_name=model_name,
                layers_to_analyze=layers_to_analyze,
                path_to_storage=path_to_storage,
                verbose=verbose
            )

    elif type_of_analysis == "delta_consecutive_layers":
        title = (f"Ranks of delta matrices with respect to the matrix in the previous layer (expl. var.: {threshold}, "
                 f"min. sing. value: {s_threshold})")

        if not file_available:
            s_dict = start_layers_delta_rank_analysis(
                model_to_analyse,
                model_name=model_name,
                layers_to_analyze=layers_to_analyze,
                path_to_storage=path_to_storage,
                verbose=verbose
            )

    elif type_of_analysis == "delta_layers_wrt_average":
        title = (f"Ranks of delta matrices with respect to the average matrix (expl. var.: {threshold}, min. sing. "
                 f"value: {s_threshold})")

        if not file_available:
            s_dict = start_layers_delta_average_rank_analysis(
                model_to_analyse,
                model_name=model_name,
                layers_to_analyze=layers_to_analyze,
                path_to_storage=path_to_storage,
                verbose=verbose
            )
    else:
        raise Exception(f"The type of analysis '{type_of_analysis}' is not supported.")

    interval = {"min": 0, "max": compute_max_possible_rank(s_dict)}
    columns_labels = [s_dict[layer_name]["labels"] for layer_name in layers_to_analyze][0]

    ranks = compute_rank(
        s_dict,
        threshold=threshold,
        s_threshold=s_threshold,
    )

    relative_ranks = {}
    if relative_plot:
        for layer_name in layers_to_analyze:
            relative_ranks[layer_name] = [
                round(rank / len(s_dict[layer_name]["s"][i]), 2) for i, rank in enumerate(ranks[layer_name])
            ]
        interval = {
            "min": 0,
            "max": 1
        }

    plot_heatmap(
        np.array(
            [ranks[layer_name] for layer_name in ranks.keys()] if not relative_plot else [relative_ranks[layer_name] for layer_name in relative_ranks.keys()]
        ).reshape(len(layers_to_analyze), -1),
        interval=interval,
        title=title,
        rows_labels=layers_to_analyze,
        columns_labels=columns_labels,
        figure_size=figure_size,
        save_path=heatmap_save_path,
        heatmap_name=f"{model_name}_{type_of_analysis}_{'_'.join(layers_to_analyze)}_{threshold}_{s_threshold}_heatmap.png"
    )


def run_mistral_7b_v0_1_analysis(
        path_to_storage: str,
        path_to_heatmap: str,
        thresholds: list = [0.99],
        s_thresholds: list = [0]
):

    if len(thresholds) != len(s_thresholds):
        raise Exception("The length of the thresholds and s_thresholds lists must be the same.")

    model_id = "mistralai/Mistral-7B-v0.1"

    layers_to_analyze_attention = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    )
    layers_to_analyze_dense = (
        "gate_proj",
        "up_proj",
        "down_proj"
    )

    types_of_analysis = [
        "original_layers",
        "delta_consecutive_layers",
        "delta_layers_wrt_average"
    ]

    for index in range(len(thresholds)):
        for type_of_analysis in types_of_analysis:
            start_layers_analysis(
                type_of_analysis,
                model_id,
                layers_to_analyze=layers_to_analyze_attention,
                threshold=thresholds[index],
                s_threshold=s_thresholds[index],
                path_to_storage=path_to_storage,
                figure_size=(24, 5),
                heatmap_save_path=path_to_heatmap,
                verbose=True
            )

            start_layers_analysis(
                type_of_analysis,
                model_id,
                layers_to_analyze=layers_to_analyze_dense,
                threshold=thresholds[index],
                s_threshold=s_thresholds[index],
                path_to_storage=path_to_storage,
                figure_size=(24, 4),
                heatmap_save_path=path_to_heatmap,
                verbose=True
            )


def run_gemma_2b_analysis(
        path_to_storage: str,
        path_to_heatmap: str,
        thresholds: list = [0.99],
        s_thresholds: list = [0]
):

    if len(thresholds) != len(s_thresholds):
        raise Exception("The length of the thresholds and s_thresholds lists must be the same.")

    model_id = "google/gemma-2b"

    layers_to_analyze_attention = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    )
    layers_to_analyze_dense = (
        "gate_proj",
        "up_proj",
        "down_proj"
    )

    types_of_analysis = [
        "original_layers",
        "delta_consecutive_layers",
        "delta_layers_wrt_average"
    ]

    for index in range(len(thresholds)):
        for type_of_analysis in types_of_analysis:
            start_layers_analysis(
                type_of_analysis,
                model_id,
                layers_to_analyze=layers_to_analyze_attention,
                threshold=thresholds[index],
                s_threshold=s_thresholds[index],
                path_to_storage=path_to_storage,
                figure_size=(14, 8),
                heatmap_save_path=path_to_heatmap,
                verbose=True
            )

            start_layers_analysis(
                type_of_analysis,
                model_id,
                layers_to_analyze=layers_to_analyze_dense,
                threshold=thresholds[index],
                s_threshold=s_thresholds[index],
                path_to_storage=path_to_storage,
                figure_size=(14, 8),
                heatmap_save_path=path_to_heatmap,
                verbose=True
            )


if __name__ == "__main__":
    path_to_storage = "/experiments/rank analysis"
    path_to_heatmap = "/experiments/rank analysis"
    thresholds = np.linspace(0.8, 0.99, 5).tolist()

    run_mistral_7b_v0_1_analysis(
        path_to_storage,
        path_to_heatmap,
        thresholds=thresholds,
        s_thresholds=[0]*len(thresholds)
    )
