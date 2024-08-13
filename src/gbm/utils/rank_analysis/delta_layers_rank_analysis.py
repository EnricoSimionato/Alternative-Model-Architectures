import pickle
from tqdm import tqdm

import numpy as np

import torch

from gbm.utils.printing_utils.printing_utils import Verbose
from gbm.utils.experiment_pipeline import Config

from gbm.utils.chatbot import load_original_model_for_causal_lm


from gbm.utils.rank_analysis.utils import (
    AnalysisTensorWrapper,
    AnalysisTensorDict
)

from gbm.utils.rank_analysis.utils import (
    extract_based_on_path,
    compute_max_possible_rank,

    plot_heatmap
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

    file_path = configuration.get("file_path")
    explained_variance_threshold = (
        configuration.get("explained_variance_threshold") if configuration.contains("explained_variance_threshold") else 0
    )
    singular_values_threshold = (
        configuration.get("singular_values_threshold") if configuration.contains("singular_values_threshold") else 0
    )

    if configuration.get("file_available"):
        print(f"File already exists. Loading data from '{file_path}'...")

        # Loading the data
        with open(file_path, "rb") as f:
            pre_analyzed_tensors = pickle.load(f)
    else:
        # Loading the model
        model = load_original_model_for_causal_lm(configuration)
        if verbose > Verbose.SILENT:
            print(f"Model loaded")

        # Extracting the layers to analyze
        extracted_layers = []
        extract_based_on_path(
            model,
            configuration.get("targets"),
            extracted_layers,
            black_list=configuration.get("black_list"),
            verbose=verbose
        )
        if verbose > Verbose.SILENT:
            print(f"Layers extracted")

        # Grouping the extracted layers by block
        extracted_layers_grouped_by_label = {}
        for extracted_layer in extracted_layers:
            label = extracted_layer.get_label()
            if label not in extracted_layers_grouped_by_label.keys():
                extracted_layers_grouped_by_label[label] = []

            extracted_layers_grouped_by_label[label].append(extracted_layer)
        if verbose > Verbose.SILENT:
            print(f"Layers grouped by label")

        # Computing the delta matrices
        pre_analyzed_tensors = AnalysisTensorDict()
        for label, extracted_layers in extracted_layers_grouped_by_label.items():
            delta_matrices = compute_delta_consecutive_matrices(
                extracted_layers,
                verbose=verbose
            )
            for delta_matrix in delta_matrices:
                delta_matrix.compute_singular_values()
            pre_analyzed_tensors.append_tensor(
                label,
                delta_matrices
            )

    matrix_types = pre_analyzed_tensors.get_unique_positional_keys(position=0)
    number_of_blocks = len(pre_analyzed_tensors.get_tensor_list(matrix_types[0]))
    ranks = np.zeros((len(matrix_types), number_of_blocks))
    relative_ranks = np.zeros((len(matrix_types), number_of_blocks))

    analyzed_tensors = AnalysisTensorDict()
    for index_label, label in tqdm(enumerate(matrix_types)):
        for index_block, matrix in enumerate(pre_analyzed_tensors.get_tensor_list(label)):
            rank = matrix.get_rank(explained_variance_threshold, singular_values_threshold, False)
            ranks[index_label, index_block] = rank
            rank = matrix.get_rank(explained_variance_threshold, singular_values_threshold, True)
            relative_ranks[index_label, index_block] = rank

            analyzed_tensors.append_tensor(
                label,
                matrix
            )

    # Saving the matrix wrappers of the layers used to perform the analysis
    with open(file_path, "wb") as f:
        pickle.dump(
            analyzed_tensors,
            f
        )
    if verbose > Verbose.SILENT:
        print(f"Data saved")

    # Plotting the results
    heatmap_name = configuration.get("heatmap_name") if configuration.contains("heatmap_name") else "heatmap"
    heatmap_name += "_expvar_" + str(explained_variance_threshold).replace('.', '_')
    plot_heatmap(
        ranks,
        interval={"min": 0, "max": compute_max_possible_rank(analyzed_tensors)},
        title="Rank analysis of the difference between consecutive matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Layer type",
        columns_labels=list(range(number_of_blocks)),
        rows_labels=matrix_types,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (10, 24),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )

    heatmap_name += "_relative"
    plot_heatmap(
        relative_ranks,
        interval={"min": 0, "max": 1},
        title="Relative rank analysis of the difference between consecutive matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Layer type",
        columns_labels=list(range(number_of_blocks)),
        rows_labels=matrix_types,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (10, 24),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )


# Definition of the functions to start the layers analysis of the delta matrices between layers and the average matrix

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

    file_path = configuration.get("file_path")
    explained_variance_threshold = (
        configuration.get("explained_variance_threshold") if configuration.contains(
            "explained_variance_threshold") else 0
    )
    singular_values_threshold = (
        configuration.get("singular_values_threshold") if configuration.contains("singular_values_threshold") else 0
    )

    if configuration.get("file_available"):
        print(f"File already exists. Loading data from '{file_path}'...")

        # Loading the data
        with open(file_path, "rb") as f:
            pre_analyzed_tensors = pickle.load(f)
    else:
        # Loading the model
        model = load_original_model_for_causal_lm(configuration)
        if verbose > Verbose.SILENT:
            print(f"Model loaded")

        # Extracting the layers to analyze
        extracted_layers = []
        extract_based_on_path(
            model,
            configuration.get("targets"),
            extracted_layers,
            black_list=configuration.get("black_list"),
            verbose=verbose
        )
        if verbose > Verbose.SILENT:
            print(f"Layers extracted")

        # Grouping the extracted layers by block
        extracted_layers_grouped_by_label = {}
        for extracted_layer in extracted_layers:
            label = extracted_layer.get_label()
            if label not in extracted_layers_grouped_by_label.keys():
                extracted_layers_grouped_by_label[label] = []

            extracted_layers_grouped_by_label[label].append(extracted_layer)
        if verbose > Verbose.SILENT:
            print(f"Layers grouped by label")

        # Computing the delta matrices
        pre_analyzed_tensors = AnalysisTensorDict()
        for label, extracted_layers in extracted_layers_grouped_by_label.items():
            delta_matrices = compute_delta_wrt_average_matrices(
                extracted_layers,
                verbose=verbose
            )
            for delta_matrix in delta_matrices:
                delta_matrix.compute_singular_values()
            pre_analyzed_tensors.append_tensor(
                label,
                delta_matrices
            )

    matrix_types = pre_analyzed_tensors.get_unique_positional_keys(position=0)
    number_of_blocks = len(pre_analyzed_tensors.get_tensor_list(matrix_types[0]))
    ranks = np.zeros((len(matrix_types), number_of_blocks))
    relative_ranks = np.zeros((len(matrix_types), number_of_blocks))

    analyzed_tensors = AnalysisTensorDict()
    for index_label, label in tqdm(enumerate(matrix_types)):
        for index_block, matrix in enumerate(pre_analyzed_tensors.get_tensor_list(label)):
            rank = matrix.get_rank(explained_variance_threshold, singular_values_threshold, False)
            ranks[index_label, index_block] = rank
            rank = matrix.get_rank(explained_variance_threshold, singular_values_threshold, True)
            relative_ranks[index_label, index_block] = rank

            analyzed_tensors.append_tensor(
                label,
                matrix
            )

    # Saving the matrix wrappers of the layers used to perform the analysis
    with open(file_path, "wb") as f:
        pickle.dump(
            analyzed_tensors,
            f
        )
    if verbose > Verbose.SILENT:
        print(f"Data saved")

    # Plotting the results
    heatmap_name = configuration.get("heatmap_name") if configuration.contains("heatmap_name") else "heatmap"
    heatmap_name += "_expvar_" + str(explained_variance_threshold).replace('.', '_')
    plot_heatmap(
        ranks,
        interval={"min": 0, "max": compute_max_possible_rank(analyzed_tensors)},
        title="Rank analysis of the difference between consecutive matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Layer type",
        columns_labels=list(range(number_of_blocks)),
        rows_labels=matrix_types,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (10, 24),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )

    heatmap_name += "_relative"
    plot_heatmap(
        relative_ranks,
        interval={"min": 0, "max": 1},
        title="Relative rank analysis of the difference between consecutive matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Layer type",
        columns_labels=list(range(number_of_blocks)),
        rows_labels=matrix_types,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (10, 24),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )
