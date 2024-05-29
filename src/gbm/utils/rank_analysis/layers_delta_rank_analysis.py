import os
import pickle

import numpy as np

import torch.nn as nn

from transformers import AutoModelForCausalLM

from gbm.utils.rank_analysis.utils import (
    compute_rank,
    extract,
    compute_max_possible_rank,

    plot_heatmap, compute_singular_values
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


def compute_delta_wrt_average_matrices(
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


def start_original_layers_rank_analysis(
        model: nn.Module,
        model_name: str,
        layers_to_analyze: tuple,
        path_to_storage: str,
        verbose: bool = False
):
    """
    Computes and stores the singular values of the layers.

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

    s_layers = {}
    for layer_name in layers_to_analyze:
        extracted_matrices = []
        extract(
            model,
            [layer_name],
            extracted_matrices
        )
        s_layers[layer_name] = {
            "s": [],
            "labels": []
        }

        for extracted_matrix in extracted_matrices:
            s = compute_singular_values(
                extracted_matrix["weight"]
            )
            s_layers[layer_name]["s"].append(s)
            s_layers[layer_name]["labels"].append(extracted_matrix["label"])
            if verbose:
                print("Singular values for", layer_name, extracted_matrix["label"], "extracted")

    if not os.path.exists(os.path.join(path_to_storage, model_name)):
        os.makedirs(os.path.join(path_to_storage, model_name))
    with open(
            os.path.join(path_to_storage, model_name, "_".join(["original_layers", "_".join(layers_to_analyze)])),
            'wb'
    ) as f:
        pickle.dump(s_layers, f)

    return s_layers


def start_layers_delta_rank_analysis(
        model: nn.Module,
        model_name: str,
        layers_to_analyze: tuple,
        path_to_storage: str,
        verbose: bool = False
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
                delta_matrix["delta_matrix"]
            )
            s_delta_layers[layer_name]["s"].append(s)
            s_delta_layers[layer_name]["labels"].append(delta_matrix["delta_label"])
            if verbose:
                print("Singular values for", layer_name, delta_matrix['delta_label'], "extracted")

    if not os.path.exists(os.path.join(path_to_storage, model_name)):
        os.makedirs(os.path.join(path_to_storage, model_name))
    with open(
            os.path.join(path_to_storage, model_name, "_".join(["delta_consecutive_layers", "_".join(layers_to_analyze)])),
            'wb'
    ) as f:
        pickle.dump(s_delta_layers, f)

    return s_delta_layers


def start_layers_delta_average_rank_analysis(
        model: nn.Module,
        model_name: str,
        layers_to_analyze: tuple,
        path_to_storage: str,
        verbose: bool = False
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
        verbose (bool):
            Whether to print some additional information.
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
                delta_matrix["delta_matrix"]
            )
            s_delta_layers_wrt_average[layer_name]["s"].append(s)
            if verbose:
                print("Singular values for", layer_name, delta_matrix['delta_label'], "extracted")

    if not os.path.exists(os.path.join(path_to_storage, model_name)):
        os.makedirs(os.path.join(path_to_storage, model_name))
    with open(
            os.path.join(path_to_storage, model_name, "_".join(["delta_layers_wrt_average", "_".join(layers_to_analyze)])),
            'wb'
    ) as f:
        pickle.dump(s_delta_layers_wrt_average, f)

    return s_delta_layers_wrt_average


def check_path_to_storage(
        path_to_storage: str,
        model_name: str,
        type_of_analysis: str,
        layers_to_analyze: tuple
) -> bool:
    """
    Checks if the path to the storage exists.

    Args:
        path_to_storage (str):
            The path to the storage.
        model_name (str):
            The name of the model.
        type_of_analysis (str):
            The type of analysis.
        layers_to_analyze (tuple):
            The layers to analyze.

    Returns:
        bool:
            Whether the path to the storage exists.
    """

    exists_directory_path = os.path.exists(
        os.path.join(
            path_to_storage, model_name
        )
    )

    if exists_directory_path:
        names_to_be_contained = [
            model_name,
            type_of_analysis
        ]
        for layer_name in layers_to_analyze:
            names_to_be_contained.append(layer_name)

        try:
            files_and_dirs = os.listdir(
                os.path.join(
                    path_to_storage, model_name
                )
            )

            files = [f for f in files_and_dirs if os.path.isfile(os.path.join(path_to_storage, model_name, f))]
            print(files)
            for file_name in files:
                elements = file_name.split("_")
                names_contained = all(element in file_name for element in elements)
                if names_contained:
                    return True

        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    return False


def start_layers_analysis(
        type_of_analysis: str,
        model: nn.Module = None,
        model_name: str = None,
        layers_to_analyze: tuple = None,
        threshold: float = 0.9,
        s_threshold: float = 0,
        figure_size: tuple = (10, 8),
        path_to_storage: str = None,
        verbose: bool = False
):

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
    else:
        print("Data storage to use for the analysis found.")

        # Load the data from the storage
        with open(file_path, 'rb') as f:
            s_dict = pickle.load(f)

    if type_of_analysis == "original_layers":
        title = "Ranks of matrices of the layers"

        if len(s_dict) == 0:
            s_dict = start_original_layers_rank_analysis(
                model,
                model_name=model_name,
                layers_to_analyze=layers_to_analyze,
                path_to_storage=path_to_storage,
                verbose=verbose
            )

    elif type_of_analysis == "delta_consecutive_layers":
        title = "Ranks of delta matrices with respect to the matrix in the previous layer"

        if len(s_dict) == 0:
            s_dict = start_layers_delta_rank_analysis(
                model,
                model_name=model_name,
                layers_to_analyze=layers_to_analyze,
                path_to_storage=path_to_storage,
                verbose=verbose
            )

    elif type_of_analysis == "delta_layers_wrt_average":
        title = "Ranks of delta matrices with respect to the average matrix"

        if len(s_dict) == 0:
            s_dict = start_layers_delta_average_rank_analysis(
                model,
                model_name=model_name,
                layers_to_analyze=layers_to_analyze,
                path_to_storage=path_to_storage,
                verbose=verbose
            )
    else:
        raise Exception(f"The type of analysis '{type_of_analysis}' is not supported.")

    columns_labels = [s_dict[layer_name]["labels"] for layer_name in layers_to_analyze]

    for layer_name in layers_to_analyze:
        ranks = compute_rank(
            s_dict,
            threshold=threshold,
            s_threshold=s_threshold
        )

    plot_heatmap(
        np.array(ranks).reshape(len(layers_to_analyze), -1),
        interval={"min": 0, "max": compute_max_possible_rank(s_dict)},
        figure_title=title,
        rows_labels=layers_to_analyze,
        columns_labels=columns_labels,
        figure_size=figure_size
    )


if __name__ == "__main__":
    #model_id = "mistralai/Mistral-7B-v0.1"
    #model_name = "Mistral-7B-v0.1"
    model_id = "google/gemma-2b"
    model_name = "gemma-2b"
    model_to_analyse = AutoModelForCausalLM.from_pretrained(model_id)
    print(model_to_analyse)
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
    
    """
    start_original_layers_rank_analysis(
        model_to_analyse,
        model_name,
        layers_to_analyze=layers_to_analyze_attention,
        path_to_storage="/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis",
        verbose=True
    )

    start_layers_delta_rank_analysis(
        model_to_analyse,
        model_name,
        layers_to_analyze=layers_to_analyze_attention,
        path_to_storage="/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis",
        verbose=True
    )

    start_layers_delta_average_rank_analysis(
        model_to_analyse,
        model_name,
        layers_to_analyze=layers_to_analyze_attention,
        path_to_storage="/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis",
        verbose=True
    )
    """
    start_original_layers_rank_analysis(
        model_to_analyse,
        model_name,
        layers_to_analyze=layers_to_analyze_dense,
        path_to_storage="/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis",
        verbose=True
    )

    start_layers_delta_rank_analysis(
        model_to_analyse,
        model_name,
        layers_to_analyze=layers_to_analyze_dense,
        path_to_storage="/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis",
        verbose=True
    )

    start_layers_delta_average_rank_analysis(
        model_to_analyse,
        model_name,
        layers_to_analyze=layers_to_analyze_dense,
        path_to_storage="/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis",
        verbose=True
    )
    #"""
