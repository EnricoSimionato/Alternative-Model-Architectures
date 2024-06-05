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
                "matrix": delta_matrix,
                "label": str(minuend_matrices[i]["label"]) + " - " + str(subtrahend_matrices[i]["label"])
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
                delta_matrix["matrix"]
            )
            s_delta_layers[layer_name]["s"].append(s)
            s_delta_layers[layer_name]["labels"].append(delta_matrix["label"])
            if verbose:
                print("Singular values for", layer_name, delta_matrix['label'], "extracted")

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
                delta_matrix["matrix"]
            )
            s_delta_layers_wrt_average[layer_name]["s"].append(s)
            s_delta_layers_wrt_average[layer_name]["labels"].append(delta_matrix["label"])
            if verbose:
                print("Singular values for", layer_name, delta_matrix['label'], "extracted")

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
) -> tuple[bool, str]:
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

            for file_name in files:
                names_contained = all(string in file_name for string in names_to_be_contained)
                if names_contained:
                    return True, os.path.join(
                        path_to_storage, model_name, file_name
                    )

        except Exception as e:
            print(f"An error occurred: {e}")
            return False, ""

    return False, ""


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
        verbose: bool = False,
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
        verbose (bool, optional):
            Whether to print some additional information. Defaults to False.
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
        title = f"Ranks of delta matrices with respect to the matrix in the previous layer (expl. var.: {threshold}, min. sing. value: {s_threshold})"

        if not file_available:
            s_dict = start_layers_delta_rank_analysis(
                model_to_analyse,
                model_name=model_name,
                layers_to_analyze=layers_to_analyze,
                path_to_storage=path_to_storage,
                verbose=verbose
            )

    elif type_of_analysis == "delta_layers_wrt_average":
        title = f"Ranks of delta matrices with respect to the average matrix (expl. var.: {threshold}, min. sing. value: {s_threshold})"

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
    path_to_storage = "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis"
    path_to_heatmap = "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis"
    thresholds = np.linspace(0.8, 0.99, 5).tolist()

    run_mistral_7b_v0_1_analysis(
        path_to_storage,
        path_to_heatmap,
        thresholds=thresholds,
        s_thresholds=[0]*len(thresholds)
    )
"""
if __name__ == "__main__":
    path = "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/rank_analysis/Mistral-7B-v0.1/delta_layers_wrt_average_gate_proj_up_proj_down_proj"

    with open(path, 'rb') as f:
        s_dict = pickle.load(f)
        print(f"Data loaded from '{path}'")

    for key in s_dict.keys():
        print(key)
        print(len(s_dict[key]["s"]))
        for s in s_dict[key]["s"]:
            print(len(s))
        print()
"""