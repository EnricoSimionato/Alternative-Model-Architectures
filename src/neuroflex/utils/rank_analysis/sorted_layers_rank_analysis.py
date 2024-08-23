import os
import pickle
from tqdm import tqdm

import numpy as np

import torch

from neuroflex.utils.printing_utils.printing_utils import Verbose
from neuroflex.utils.experiment_pipeline.config import Config, get_path_to_configurations
from neuroflex.utils.chatbot.conversation_utils import load_original_model_for_causal_lm

from neuroflex.utils.rank_analysis.delta_layers_rank_analysis import compute_delta_matrices

from neuroflex.utils.rank_analysis.utils import (
    AnalysisTensorWrapper,
    AnalysisTensorDict,
)

from neuroflex.utils.rank_analysis.utils import (
    extract_based_on_path,
    compute_max_possible_rank,

    plot_heatmap
)


def compute_cosine(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = 0,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        x (torch.Tensor):
            The first tensor.
        y (torch.Tensor):
            The second tensor.
        dim (int):
            The dimension to compute the cosine similarity. Default: 0.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        torch.Tensor:
            The cosine similarity between the two tensors.
    """

    if verbose > Verbose.INFO:
        print(f"Computing cosine similarity.")
        print(f"The shape of x is {x.shape}.")
        print(f"The shape of y is {y.shape}.")
        print(f"The cosine similarity between x and y is "
              f"{torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()}")

    return torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=dim)


def compute_similarity(
        x: torch.Tensor,
        y: torch.Tensor,
        similarity_type: str = "cosine",
        verbose: Verbose = Verbose.INFO,
        **kwargs
) -> torch.Tensor:
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
        **kwargs:
            Additional arguments for the similarity computation.

    Returns:
        torch.Tensor:
            The similarity between the two tensors.
    """

    if similarity_type == "cosine":
        return compute_cosine(x, y, dim=kwargs["dim"], verbose=verbose)
    else:
        raise Exception(f"Similarity type '{similarity_type}' not supported.")


def sort_columns(
        tensor: torch.Tensor,
        list_of_indices: list,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Sorts the columns of a matrix based on a list of indices.

    Args:
        tensor (torch.Tensor):
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

    sorted_tensor = tensor[:, list_of_indices]

    return sorted_tensor


def sort_rows(
        tensor: torch.Tensor,
        list_of_indices: list,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Sorts the rows of a matrix based on a list of indices.

    Args:
        tensor (torch.Tensor):
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

    sorted_tensor = tensor[list_of_indices, :]

    return sorted_tensor


def compute_indices_sorting(
        layers_in_block_1: list[AnalysisTensorWrapper],
        layers_in_block_2: list[AnalysisTensorWrapper],
        layer_index_for_similarity_1: int,
        layer_index_for_similarity_2: int,
        axis: int,
        similarity_type: str = "cosine",
        verbose: Verbose = Verbose.INFO
) -> [list, dict[torch.Tensor]]:
    """
    Computes the indices for the sorting of the elements of the layers in block 2 to be as close as possible to the
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
        dict[torch.Tensor]:
            The matrix of similarities between the layers in block 1 and block 2.


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
            ).item()
    similarities_copy = similarities.clone()

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

    return ordering, similarities_copy


def sort_elements_in_layers(
        list_of_subsequent_matrices: list[AnalysisTensorWrapper],
        list_of_indices: list,
        axes: list,
        verbose: Verbose = Verbose.INFO
) -> list:
    """
    Sorts the elements in the layers based on a list of indices.

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
            sorted_tensor = sort_rows(layer.get_tensor(), list_of_indices, verbose=verbose)
        elif axes[index] == 1:
            sorted_tensor = sort_columns(layer.get_tensor(), list_of_indices, verbose=verbose)
        else:
            raise Exception(f"Axis '{axes[index]}' not supported.")

        if verbose > Verbose.INFO:
            print(f"Original tensor: {layer.get_tensor()[0, :10]}")
            print(f"Indices: {list_of_indices[:10]}")
            print(f"Sorted tensor: {sorted_tensor[0, :10]}")

        reordered_matrices.append(
            AnalysisTensorWrapper(
                sorted_tensor,
                name=layer.get_name(),
                label=layer.get_label(),
                path=layer.get_path(),
                block_index=layer.get_block_index(),
                layer=layer.get_layer()
            )
        )

    return reordered_matrices


def perform_sorted_layers_rank_analysis(
        configuration: Config,
        verbose: Verbose = Verbose.INFO
) -> None:
    """
    Performs the aligned layers rank analysis.

    Args:
        configuration (Config):
            The configuration of the experiment.
        verbose (Verbose, optional):
            The verbosity level. Default is Verbose.INFO.
    """

    file_path = configuration.get("file_path")
    similarities_path = os.path.join(configuration.get("directory_path"), "similarities")
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
        if verbose > Verbose.INFO:
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

        similarity_guide_name = configuration.get("similarity_guide_name")
        pre_analyzed_tensors = AnalysisTensorDict()
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
                    ordering, similarities = compute_indices_sorting(
                        layers_in_block_1,
                        layers_in_block_2,
                        layer_index_for_similarity_1=similarity_guide_index,
                        layer_index_for_similarity_2=similarity_guide_index,
                        axis=1 if similarity_guide_index == len(layers_in_block_1) - 1 else 0,
                        verbose=verbose
                    )
                    if verbose > Verbose.INFO:
                        print(f"\nIndices computed {ordering[:10]}")

                    print(f"Block {block_index_1} and block {block_index_2} have mean (on the rows) highest similarity "
                          f"{torch.mean(torch.max(similarities, dim=1).values).item()}.")
                    print(f"Block {block_index_1} and block {block_index_2} have mean (on the columns) highest "
                          f"similarity {torch.mean(torch.max(similarities, dim=0).values).item()}.")
                    # Saving the similarities
                    try:
                        with open(similarities_path, 'rb') as file:
                            computed_similarities = pickle.load(file)
                    except FileNotFoundError:
                        computed_similarities = {}
                    computed_similarities.update({(block_index_1, block_index_2): similarities})
                    with open(similarities_path, 'wb') as f:
                        pickle.dump(computed_similarities, f)

                    # Using the ordering to sort the vectors in the matrices of block 2
                    sorted_layers_in_block_2 = sort_elements_in_layers(
                        layers_in_block_2,
                        ordering,
                        axes=[
                            1 if index == len(layers_in_block_2) - 1 else 0
                            for index in range(len(layers_in_block_2))
                        ],
                        verbose=verbose
                    )

                    if verbose >= Verbose.DEBUG:
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
                    pre_analyzed_tensors.append_tensor(
                        key,
                        delta_matrices
                    )

    try:
        with open(similarities_path, 'rb') as file:
            computed_similarities = pickle.load(file)
            for key, similarities in computed_similarities.items():
                block_index_1, block_index_2 = key
                print(f"Block {block_index_1} and block {block_index_2} have mean (on the rows) highest similarity "
                      f"{torch.mean(torch.max(similarities, dim=1).values).item()}.")
                print(f"Block {block_index_1} and block {block_index_2} have mean (on the columns) highest "
                      f"similarity {torch.mean(torch.max(similarities, dim=0).values).item()}.")
    except FileNotFoundError:
        print("No file containing the similarities found.")

    # Retrieving the indices of the blocks of the model
    blocks_indexes_1 = pre_analyzed_tensors.get_unique_positional_keys(position=0, sort=True)
    ranks = np.zeros((len(blocks_indexes_1), len(blocks_indexes_1)))
    relative_ranks = np.zeros((len(blocks_indexes_1), len(blocks_indexes_1)))

    # Performing the rank analysis of the difference between the matrices of the model aligned based on the similarity
    analyzed_tensors = AnalysisTensorDict()
    for block_index_1 in tqdm(blocks_indexes_1):
        filtered_dictionary_all_delta_matrices = pre_analyzed_tensors.filter_by_positional_key(
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
                rank = delta_matrix.get_rank(explained_variance_threshold, singular_values_threshold, False)
                ranks[int(block_index_1), int(block_index_2)] += rank
                rank = delta_matrix.get_rank(explained_variance_threshold, singular_values_threshold, True)
                relative_ranks[int(block_index_1), int(block_index_2)] += rank

                print(f"The rank of the difference matrix of the blocks {block_index_1} and {block_index_2} with label "
                      f"{delta_matrix.get_label()} and path {delta_matrix.get_path()} is {rank}.")
            ranks[int(block_index_1), int(block_index_2)] /= len(delta_matrices)
            relative_ranks[int(block_index_1), int(block_index_2)] /= len(delta_matrices)
            print(f"The average rank of the difference matrices of the blocks {block_index_1} and {block_index_2} is "
                  f"{ranks[int(block_index_1), int(block_index_2)]}.\n")

            analyzed_tensors.append_tensor(
                key,
                delta_matrices
            )

    # Saving the matrix wrappers of the layers used to perform the analysis
    with open(file_path, "wb") as f:
        pickle.dump(analyzed_tensors, f)

    # Plotting the results
    heatmap_name = configuration.get("heatmap_name") if configuration.contains("heatmap_name") else "heatmap"
    heatmap_name += "_expvar_" + str(explained_variance_threshold).replace('.', '_')
    plot_heatmap(
        ranks,
        interval={"min": 0, "max": compute_max_possible_rank(analyzed_tensors)},
        title="Average rank analysis of the difference between matrices of the model aligned based on the similarity" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Block indexes",
        columns_labels=blocks_indexes_1,
        rows_labels=blocks_indexes_1,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (26, 26),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )

    heatmap_name += "_relative"
    plot_heatmap(
        relative_ranks,
        interval={"min": 0, "max": 1},
        title="Average relative rank analysis of the difference between matrices of the model aligned based on the similarity" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Block indexes",
        columns_labels=blocks_indexes_1,
        rows_labels=blocks_indexes_1,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (26, 26),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )
