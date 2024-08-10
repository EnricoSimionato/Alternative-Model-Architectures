import os
import pickle
import sys
from tqdm import tqdm

import numpy as np

import torch

from gbm.utils.experiment_pipeline.experiment import get_path_to_experiments
from gbm.utils.printing_utils.printing_utils import Verbose
from gbm.utils.experiment_pipeline.config import Config, get_path_to_configurations
from gbm.utils.chatbot.conversation_utils import load_original_model_for_causal_lm

from gbm.utils.rank_analysis.layers_delta_rank_analysis import compute_delta_matrices

from gbm.utils.rank_analysis.utils import (
    AnalysisTensorWrapper,
    AnalysisTensorDict
)

from gbm.utils.rank_analysis.utils import (
    extract_based_on_path,
    plot_heatmap
)


def compute_cosine(
        x: torch.Tensor,
        y: torch.Tensor,
        verbose: Verbose = Verbose.INFO
) -> float:
    """
    Computes the cosine similarity between two tensors.

    Args:
        x (torch.Tensor):
            The first tensor.
        y (torch.Tensor):
            The second tensor.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        float:
            The cosine similarity between the two tensors.
    """

    if verbose > Verbose.INFO:
        print(f"Computing cosine similarity.")
        print(f"The shape of x is {x.shape}.")
        print(f"The shape of y is {y.shape}.")
        print(f"The cosine similarity between x and y is "
              f"{torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()}")

    return torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()


def compute_similarity(
        x: torch.Tensor,
        y: torch.Tensor,
        similarity_type: str = "cosine",
        verbose: Verbose = Verbose.INFO
) -> float:
    """
    Compute the similarity between two tensors.

    Args:
        x (torch.Tensor):
            The first tensor.
        y (torch.Tensor):
            The second tensor.
        similarity_type (str):
            The type of similarity to compute. Default: "cosine".
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        float:
            The similarity between the two tensors.
    """

    if similarity_type == "cosine":
        return compute_cosine(x, y, verbose=verbose)
    else:
        raise Exception(f"Similarity type '{similarity_type}' not supported.")


def align_columns(
        matrix: torch.Tensor,
        list_of_indices: list,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Aligns the columns of a matrix based on a list of indices.

    Args:
        matrix (torch.Tensor):
            The matrix.
        list_of_indices (list):
            The list of indices to align the columns.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        torch.Tensor:
            The matrix with the columns aligned.
    """

    if verbose >= Verbose.DEBUG:
        print(f"Sorting the columns")

    sorted_matrix = matrix[:, list_of_indices]

    return sorted_matrix


def align_rows(
        matrix: torch.Tensor,
        list_of_indices: list,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Aligns the rows of a matrix based on a list of indices.

    Args:
        matrix (torch.Tensor):
            The matrix.
        list_of_indices (list):
            The list of indices to align the rows.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        torch.Tensor:
            The matrix with the rows aligned.
    """

    if verbose >= Verbose.DEBUG:
        print(f"Sorting the rows")

    sorted_matrix = matrix[list_of_indices, :]

    return sorted_matrix


def compute_indices_alignment(
        layers_in_block_1: list[AnalysisTensorWrapper],
        layers_in_block_2: list[AnalysisTensorWrapper],
        layer_index_for_similarity_1: int,
        layer_index_for_similarity_2: int,
        axis: int,
        similarity_type: str = "cosine",
        verbose: Verbose = Verbose.INFO
) -> list:
    """
    Computes the indices for the alignment of the elements of the layers in block 2 to be as close as possible to the
    layers in block 1.

    Args:
        layers_in_block_1 (list[AnalysisTensorWrapper]):
            The list of layers in block 1.
        layers_in_block_2 (list[AnalysisTensorWrapper]):
            The list of layers in block 2.
        layer_index_for_similarity_1 (int):
            The index of the layer in block 1 to use for the similarity computation.
        layer_index_for_similarity_2 (int):
            The index of the layer in block 2 to use for the similarity computation.
        axis (int):
            The axis to align the elements.
        similarity_type (str):
            The type of similarity to use to align. Default: "cosine".
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        list:
            The list of indices to reorder the layers in block 2.

    Raises:
        Exception:
            If the axis is not supported.
    """

    layer_for_similarity_1 = layers_in_block_1[layer_index_for_similarity_1].get_tensor()
    layer_for_similarity_2 = layers_in_block_2[layer_index_for_similarity_2].get_tensor()

    if axis == 0:
        pass
    elif axis == 1:
        layer_for_similarity_1 = layer_for_similarity_1.transpose(0, 1)
        layer_for_similarity_2 = layer_for_similarity_2.transpose(0, 1)
    else:
        raise Exception(f"Axis '{axis}' not supported.")

    similarities = torch.zeros(layer_for_similarity_1.shape[0], layer_for_similarity_2.shape[0])

    for i, vector_1 in enumerate(layer_for_similarity_1):
        for j, vector_2 in enumerate(layer_for_similarity_2):
            if verbose > Verbose.INFO:
                print(f"Computing similarity between "
                      f"{'row' if axis == 0 else 'column' if axis == 1 else '???'} {i} and "
                      f"{'row' if axis == 0 else 'column' if axis == 1 else '???'} {j}.")
                print()
            similarities[i, j] = compute_similarity(
                vector_1,
                vector_2,
                similarity_type=similarity_type,
                verbose=verbose
            )

    # Computing the ordering of the vectors of the matrices in the second block to have the maximum possible similarity
    ordering = []
    for similarity_row in similarities.transpose(0, 1):
        minimum = torch.min(similarity_row)
        while torch.argmax(similarity_row) in ordering:
            if verbose > Verbose.INFO:
                indexed_values = list(enumerate(similarity_row))
                sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1])
                print("Sorted Values:", [index for index, value in sorted_indexed_values[:10]])
                print("Sorted Indices:",  [value for index, value in sorted_indexed_values[:10]])

            similarity_row[torch.argmax(similarity_row)] = minimum - 1

        ordering.append(torch.argmax(similarity_row))

        if verbose > Verbose.INFO:
            print()

    if verbose > Verbose.INFO:
        print(f"New ordering: {ordering}")

    return ordering


def align_elements_in_layers(
        list_of_subsequent_matrices: list[AnalysisTensorWrapper],
        list_of_indices: list,
        axes: list,
        verbose: Verbose = Verbose.INFO
) -> list:
    """
    Aligns the elements in the layers based on a list of indices.

    Args:
        list_of_subsequent_matrices (list):
            The list of subsequent matrices.
        list_of_indices (list):
            The list of the lists of indices to align the elements.
        axes (list):
            The list of axes to align the elements.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        list:
            The aligned matrices.

    Raises:
        Exception:
            If the axis is not supported.
    """

    reordered_matrices = []

    for index in range(len(list_of_subsequent_matrices)):
        layer = list_of_subsequent_matrices[index]
        if axes[index] == 0:
            aligned_tensor = align_rows(layer.get_tensor(), list_of_indices, verbose=verbose)
        elif axes[index] == 1:
            aligned_tensor = align_columns(layer.get_tensor(), list_of_indices, verbose=verbose)
        else:
            raise Exception(f"Axis '{axes[index]}' not supported.")

        if verbose > Verbose.INFO:
            print(f"Original tensor: {layer.get_tensor()[0, :10]}")
            print(f"Indices: {list_of_indices[:10]}")
            print(f"Sorted tensor: {aligned_tensor[0, :10]}")

        reordered_matrices.append(
            AnalysisTensorWrapper(
                aligned_tensor,
                name=layer.get_name(),
                label=layer.get_label(),
                path=layer.get_path(),
                block_index=layer.get_block_index(),
                layer=layer.get_layer()
            )
        )

    return reordered_matrices


def perform_aligned_layers_rank_analysis(
        configuration: Config,
        paths_layers_to_analyze: list = (),
        black_list: list = (),
        similarity_guide_name: int = "output_dense",
        explained_variance_threshold: float = 0.9,
        singular_values_threshold: float = 0,
        relative_plot: bool = True,
        precision: int = 2,
        figure_size: tuple = (24, 5),
        path_to_storage: str = None,
        verbose: Verbose = Verbose.INFO
) -> None:
    """
    Performs the aligned layers rank analysis.

    Args:
        configuration (Config):
            The configuration of the experiment.
        paths_layers_to_analyze (list, optional):
            The paths to the layers to analyze.
        black_list (list, optional):
            The list of layers to exclude from the analysis.
        similarity_guide_name (str, optional):
            The name of the layer to use as a guide for the similarity.
        explained_variance_threshold (float, optional):
            The threshold to consider the rank of the layers.
        singular_values_threshold (float, optional):
            The threshold to consider the singular values.
        relative_plot (bool, optional):
            Whether to plot the relative plot.
        precision (int, optional):
            The precision of the plot.
        figure_size (tuple, optional):
            The size of the figure.
        path_to_storage (str, optional):
            The path to the storage.
        verbose (Verbose, optional):
            The verbosity level.
    """

    # Checking if the path to the storage exists
    model_name = configuration.get("original_model_id").split("/")[-1]
    words_to_be_in_the_file_name = (["paths"] + paths_layers_to_analyze +
                                    ["black_list"] + black_list +
                                    ["guide"] + [similarity_guide_name])
    file_available, directory_path, file_name = check_path_to_storage(
        path_to_storage,
        "aligned_layers_rank_analysis",
        model_name,
        tuple(words_to_be_in_the_file_name)
    )
    file_path = os.path.join(directory_path, file_name)

    print(f"{'File to load data available' if file_available else 'No file to load data'}")
    print(f"File path: {file_path}")

    if file_available:
        print(f"File already exists. Loading data from '{file_path}'...")

        # Loading the data
        with open(file_path, "rb") as f:
            dictionary_all_delta_matrices = pickle.load(f)
    else:
        # Loading the model
        model = load_original_model_for_causal_lm(configuration)
        if verbose > Verbose.INFO:
            print(f"Model loaded")

        # Extracting the layers to analyze
        extracted_layers = []
        extract_based_on_path(
            model,
            paths_layers_to_analyze,
            extracted_layers,
            black_list=black_list,
            verbose=verbose
        )
        if verbose > Verbose.SILENT:
            print(f"Layers extracted")

        # Grouping the extracted layers by block
        extracted_layers_grouped_by_block = {}
        for extracted_layer in extracted_layers:
            block_index = extracted_layer.get_block_index()
            if block_index not in extracted_layers_grouped_by_block.keys():
                extracted_layers_grouped_by_block[block_index] = []

            extracted_layers_grouped_by_block[block_index].append(extracted_layer)

        if verbose > Verbose.SILENT:
            print(f"Layers grouped by block")
        if verbose > Verbose.INFO:
            print(extracted_layers_grouped_by_block.keys())
            print(extracted_layers_grouped_by_block)

        # Computing the ranks of the difference of the layers in the blocks

        dictionary_all_delta_matrices = AnalysisTensorDict()
        for block_index_1 in tqdm(extracted_layers_grouped_by_block.keys()):
            for block_index_2 in extracted_layers_grouped_by_block.keys():
                if block_index_1 != block_index_2:
                    layers_in_block_1 = extracted_layers_grouped_by_block[block_index_1]
                    layers_in_block_2 = extracted_layers_grouped_by_block[block_index_2]

                    similarity_guide_index = None
                    for index, element in enumerate(layers_in_block_1):
                        if similarity_guide_name in element.get_label():
                            similarity_guide_index = index
                            break
                    if similarity_guide_index is None:
                        raise Exception(f"Layer '{similarity_guide_name}' not found in block {block_index_1}.")

                    # Sorting the layers in block 2 to match the layers in block 1 in order to be as similar as possible

                    # Computing the ordering
                    ordering = compute_indices_alignment(
                        layers_in_block_1,
                        layers_in_block_2,
                        layer_index_for_similarity_1=similarity_guide_index,
                        layer_index_for_similarity_2=similarity_guide_index,
                        axis=1 if similarity_guide_index == len(layers_in_block_1) - 1 else 0,
                        verbose=verbose
                    )
                    if verbose > Verbose.INFO:
                        print(f"\nIndices computed {ordering[:10]}")

                    # Using the ordering to sort the vectors in the matrices of block 2
                    sorted_layers_in_block_2 = align_elements_in_layers(
                        layers_in_block_2,
                        ordering,
                        axes=[1
                              if index == len(layers_in_block_2) - 1
                              else 0
                              for index in range(len(layers_in_block_2))
                              ],
                        verbose=verbose
                    )
                    if verbose > Verbose.INFO:
                        print(f"\nLayers sorted based on similarity")

                    # Computing the delta matrices between the layers in block 1 and the sorted layers in block 2
                    delta_matrices = compute_delta_matrices(
                        layers_in_block_1,
                        sorted_layers_in_block_2,
                        verbose=verbose
                    )

                    # Computing the singular values of the delta matrices of the layers in the two different blocks
                    for delta_matrix in delta_matrices:
                        delta_matrix.compute_singular_values()

                    key = (block_index_1, block_index_2)
                    dictionary_all_delta_matrices.append_tensor(
                        key,
                        delta_matrices
                    )

    # Retrieving the indices of the blocks of the model
    blocks_indexes_1 = dictionary_all_delta_matrices.get_unique_positional_keys(position=0, sort=True)
    ranks = np.zeros((len(blocks_indexes_1), len(blocks_indexes_1)))

    # Performing the rank analysis of the difference between the matrices of the model aligned based on the similarity
    dictionary_all_analyzed_matrices = AnalysisTensorDict()
    for block_index_1 in tqdm(blocks_indexes_1):
        filtered_dictionary_all_delta_matrices = dictionary_all_delta_matrices.filter_by_positional_key(
            key=block_index_1,
            position=0
        )
        blocks_indexes_2 = filtered_dictionary_all_delta_matrices.get_unique_positional_keys(position=1, sort=True)
        for block_index_2 in blocks_indexes_2:
            key = (block_index_1, block_index_2)
            delta_matrices = filtered_dictionary_all_delta_matrices.get_tensor_list(key)

            for delta_matrix in delta_matrices:
                # Computing the rank of the difference matrix given the explained variance threshold and the
                # singular values threshold
                rank = delta_matrix.get_rank(
                    explained_variance_threshold=explained_variance_threshold,
                    singular_values_threshold=singular_values_threshold,
                    relative=relative_plot
                )
                ranks[int(block_index_1), int(block_index_2)] += rank

                print(f"The rank of the difference matrix of the blocks {block_index_1} and {block_index_2} with label "
                      f"{delta_matrix.get_label()} and path {delta_matrix.get_path()} is {rank}.")
            ranks[int(block_index_1), int(block_index_2)] /= len(delta_matrices)
            print(f"The average rank of the difference matrices of the blocks {block_index_1} and {block_index_2} is "
                  f"{ranks[int(block_index_1), int(block_index_2)]}.\n")

            dictionary_all_analyzed_matrices.append_tensor(
                key,
                delta_matrices
            )

    # Saving the matrix wrappers of the layers used to perform the analysis
    with open(file_path, "wb") as f:
        pickle.dump(
            dictionary_all_analyzed_matrices,
            f
        )

    plot_heatmap(
        ranks,
        interval={"min": 0, "max": 1},
        title="Average rank analysis of the difference between matrices of the model aligned based on the similarity",
        x_title="Block indexes",
        y_title="Block indexes",
        columns_labels=blocks_indexes_1,
        rows_labels=blocks_indexes_1,
        figure_size=figure_size,
        save_path=directory_path,
        heatmap_name="heatmap",
        show=True
    )


def check_path_to_storage(
        path_to_storage: str,
        type_of_analysis: str,
        model_name: str,
        strings_to_be_in_the_name: tuple
) -> tuple[bool, str, str]:
    """
    Checks if the path to the storage exists.
    If the path exists, the method returns a positive flag and the path to the storage of the experiment data.
    If the path does not exist, the method returns a negative flag and creates the path for the experiment returning it.

    Args:
        path_to_storage (str):
            The path to the storage where the experiments data have been stored or will be stored.
        type_of_analysis (str):
            The type of analysis to be performed on the model.
        model_name (str):
            The name of the model to analyze.
        strings_to_be_in_the_name (tuple):
            The strings to be used to create the name or to find in the name of the stored data of the considered
            experiment.

    Returns:
        bool:
            A flag indicating if the path to the storage of the specific experiment already exists.
        str:
            The path to the storage of the specific experiment.
        str:
            The name of the file to store the data.

    Raises:
        Exception:
            If the path to the storage does not exist.
    """

    if not os.path.exists(path_to_storage):
        raise Exception(f"The path to the storage '{path_to_storage}' does not exist.")

    # Checking if the path to the storage of the specific experiment already exists
    exists_directory_path = os.path.exists(
        os.path.join(
            path_to_storage, model_name
        )
    ) & os.path.isdir(
        os.path.join(
            path_to_storage, model_name
        )
    ) & os.path.exists(
        os.path.join(
            path_to_storage, model_name, type_of_analysis
        )
    ) & os.path.isdir(
        os.path.join(
            path_to_storage, model_name, type_of_analysis
        )
    )

    exists_file = False
    directory_path = os.path.join(
        path_to_storage, model_name, type_of_analysis
    )
    file_name = None
    if exists_directory_path:
        try:
            files_and_dirs = os.listdir(
                directory_path
            )

            # Extracting the files
            files = [
                f
                for f in files_and_dirs
                if os.path.isfile(os.path.join(path_to_storage, model_name, type_of_analysis, f))
            ]

            # Checking if some file ame contains the required strings
            for f_name in files:
                names_contained = all(string in f_name for string in strings_to_be_in_the_name)
                if names_contained:
                    exists_file = True
                    file_name = f_name
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
            return False, "", ""

    else:
        os.makedirs(
            os.path.join(
                directory_path
            )
        )

    if not exists_file:
        file_name = "_".join(strings_to_be_in_the_name)

    return exists_file, directory_path, str(file_name)


def main():
    """
    Main method to start the aligned layers rank analysis
    """

    if len(sys.argv) < 3:
        raise Exception("Please provide the name of the configuration file and the environment.\n"
                        "Example: python aligned_layers_rank_analysis.py config_name environment"
                        "'environment' can be 'local' or 'server' or 'colab'.")

    # Extracting the configuration name and the environment
    config_name = sys.argv[1]
    environment = sys.argv[2]

    # Loading the configuration
    config = Config(
        os.path.join(get_path_to_configurations(environment), "rank_analysis", config_name)
    )

    # Starting the aligned layers rank analysis
    perform_aligned_layers_rank_analysis(
        configuration=config,
        paths_layers_to_analyze=config.get("targets"),
        black_list=config.get("black_list"),
        explained_variance_threshold=config.get("explained_variance_threshold"),
        singular_values_threshold=(
            config.get("singular_values_threshold")
            if config.contains("singular_values_threshold")
            else 0
        ),
        relative_plot=config.get("relative_plot") if config.contains("relative_plot") else True,
        precision=config.get("precision") if config.contains("precision") else 2,
        figure_size=config.get("figure_size") if config.contains("figure_size") else (24, 5),
        path_to_storage=os.path.join(get_path_to_experiments(environment), "rank_analysis"),
        verbose=Verbose(config.get("verbose")) if config.contains("verbose") else Verbose.INFO
    )


if __name__ == "__main__":
    main()
