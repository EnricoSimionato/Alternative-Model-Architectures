import os
import pickle as pkl
import logging
from tqdm import tqdm

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

from neuroflex.utils.printing_utils.printing_utils import Verbose
from neuroflex.utils.experiment_pipeline import Config

from neuroflex.utils.classification.pl_datasets import IMDBDataModule

from neuroflex.matrixplorer.sorted_layers_rank_analysis import compute_cosine
from neuroflex.utils.classification.classification_utils import (
    load_original_model_for_sequence_classification,
    load_tokenizer_for_sequence_classification,
)

from neuroflex.matrixplorer.utils import AnalysisModelWrapper


def perform_activations_analysis(
        configuration: Config,
) -> None:
    """
    Performs the activations' analysis.

    Args:
        configuration (Config):
            The configuration object containing the necessary information to perform the analysis.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name_no_format = configuration.get("file_name_no_format")

    logging.basicConfig(filename=configuration.get("log_path"), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting the activations' analysis with the configuration.")

    # Getting the parameters related to the analysis from the configuration
    verbose = configuration.get_verbose()
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (100, 20)
    heatmap_size = configuration.get("heatmap_size") if configuration.contains("heatmap_size") else (40, 40)
    num_iterations = configuration.get("num_iterations") if configuration.contains("num_iterations") else 1
    batch_size = configuration.get("batch_size") if configuration.contains("batch_size") else 64
    num_workers = configuration.get("num_workers") if configuration.contains("num_workers") else 1
    seed = configuration.get("seed") if configuration.contains("seed") else 42
    max_len = configuration.get("max_len") if configuration.contains("max_len") else 512

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        logger.info(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        logger.info(f"Data loaded from the file '{file_path}'.")
    else:
        # Loading the model
        model = load_original_model_for_sequence_classification(configuration)
        logger.info(f"Model loaded.")
        # Loading the tokenizer
        tokenizer = load_tokenizer_for_sequence_classification(configuration)
        logger.info(f"Tokenizer loaded.")
        # Wrapping the model
        model_wrapper = AnalysisModelWrapper(model, configuration.get("targets"), configuration.get("black_list") if configuration.contains("black_list") else None)
        logger.info(f"Model wrapped.")
        print(model_wrapper)

        if configuration.get("dataset_id") == "stanfordnlp/imdb":
            # Loading the dataset
            dataset = IMDBDataModule(
                tokenizer=tokenizer,
                max_len=max_len,
                batch_size=batch_size,
                num_workers=num_workers,
                split=(0.8, 0.1, 0.1),
                seed=seed
            )
            dataset.setup()
        else:
            raise Exception("The dataset is not recognized.")
        logger.info(f"Dataset loaded.")

        # Performing the activation analysis
        data_loader = dataset.train_dataloader()
        verbose.print("Staring to feed the inputs to the model.", Verbose.SILENT)
        for idx, batch in tqdm(enumerate(data_loader)):
            logger.info(f"Batch {idx + 1} out of {num_iterations}.")
            if idx == num_iterations:
                break

            #Preparing the inputs
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch.get("labels")
            }

            # Forward pass through the model
            y = model_wrapper.forward(**inputs)
        logger.info(f"All activations computed.")

        data = model_wrapper

        # Saving the activations
        logger.info(f"Storing the data for future usage.")
        with open(f"{file_path}", "wb") as f:
            pkl.dump(data, f)
        logger.info(f"Data saved to the file '{file_path}'.")

    # Extracting the activations
    model_wrapper = data
    activations = model_wrapper.get_activations()
    if verbose >= Verbose.DEBUG:
        print_nested_dictionary(activations)

    # Flattening the activations
    flattened_activations = {}
    flatten_dictionary(flattened_activations, activations)

    key_value_couples = [(activation_dict_key, activation_dict_value["mean_activations"]) for activation_dict_key, activation_dict_value in flattened_activations.items() if activation_dict_value["mean_activations"] is not None]

    # Filtering the activations
    filtered_key_value_couples = [key_value_couple for key_value_couple in key_value_couples if is_at_least_one_element_in_list(configuration.get("targets"), key_value_couple[0].split(" -> "))]

    # Printing the statistics of the activations of one layer
    mean_activations = filtered_key_value_couples[0][1]
    print(f"Shape mean activations: {mean_activations.shape}")
    print(f"Max mean activations: {mean_activations.max()}")
    print(f"Min mean activations: {mean_activations.min()}")
    print(f"Max absolute mean activations: {mean_activations.abs().max()}")
    print(f"Min absolute activations: {mean_activations.abs().min()}")
    print(f"Average mean activations: {mean_activations.mean()}")
    print(f"Variance mean activations: {mean_activations.var()}")
    print(f"Average absolute mean activations: {mean_activations.abs().mean()}")
    print(f"Variance absolute mean activations: {mean_activations.abs().var()}")

    # Plotting the mean activations
    fig_1, axis_1 = plt.subplots(1, 1, figsize=fig_size)
    fig_1.suptitle("Mean activations of different layers")
    x = range(len(key_value_couples[0][1]))
    for key_value_couple in filtered_key_value_couples:
        if len(key_value_couple[1]) != len(x):
            raise Exception("The length of the tensors is not the same for all the embeddings.")

    for key_value_couple in filtered_key_value_couples:
        axis_1.plot(key_value_couple[1].detach().numpy(), label=f"{key_value_couple[0]}")

    axis_1.set_title("Mean activations")
    axis_1.set_xlabel("Component index")
    axis_1.set_ylabel("Value of the component")
    axis_1.legend()

    # Saving the plot
    fig_path = os.path.join(directory_path, file_name_no_format + "_mean_activations.png")
    fig_1.savefig(f"{fig_path}")
    logger.info(f"Plot saved to '{fig_path}'.")

    # Computing the similarity matrix
    similarity_matrix = torch.zeros((len(filtered_key_value_couples), len(filtered_key_value_couples)))
    for index_1, key_value_1 in enumerate(filtered_key_value_couples):
        for index_2, key_value_2 in enumerate(filtered_key_value_couples):
            similarity_matrix[index_1, index_2] = compute_cosine(key_value_1[1], key_value_2[1], dim=1)

    fig_2, axis_2 = plt.subplots(1, 1, figsize=heatmap_size)
    fig_2.suptitle("Similarity matrix of the mean activations of different layers")
    heatmap = axis_2.imshow(similarity_matrix, cmap="seismic", interpolation="nearest", vmin=-1, vmax=1)

    axis_2.set_xlabel("Layer Label")
    axis_2.set_xticks(range(len(filtered_key_value_couples)))
    axis_2.set_xticklabels([key_value_couple[0] for key_value_couple in filtered_key_value_couples], rotation=90)

    axis_2.set_ylabel("Layer Label")
    axis_2.set_yticks(range(len(filtered_key_value_couples)))
    axis_2.set_yticklabels([key_value_couple[0] for key_value_couple in filtered_key_value_couples])

    # Adding the colorbar
    divider = make_axes_locatable(axis_2)
    colormap_axis = divider.append_axes("right", size="5%", pad=0.05)
    fig_2.colorbar(
        heatmap,
        cax=colormap_axis
    )
    plt.tight_layout()

    # Saving the plot
    fig_path = os.path.join(directory_path, file_name_no_format + "_similarity_matrix.png")
    fig_2.savefig(f"{fig_path}")
    logger.info(f"Plot saved to '{fig_path}'.")

    logger.info(f"Activations' analysis completed.")


def print_nested_dictionary(
        dictionary: dict,
        level: int = 0
) -> None:
    """
    Prints the nested dictionary.

    Args:
        dictionary (dict):
            The dictionary to print.
        level (int):
            The level of the dictionary.
    """

    for key, value in dictionary.items():
        if isinstance(value, dict):
            tabs = "\t" * level
            print(f"{tabs}({key}): ")
            print_nested_dictionary(value, level + 1)
        else:
            tabs = "\t" * level
            print(f"{tabs}({key}): {str(value)[:30]}")


def flatten_dictionary(
        global_dictionary: dict,
        dictionary: dict,
        path: str = ""
) -> None:
    """
    Flattens the nested dictionary.

    Args:
        dictionary (dict):
            The dictionary to flatten.
        global_dictionary (dict):
            The global dictionary to store the results.
    """

    nested_dict = False
    for key, value in dictionary.items():
        if isinstance(value, dict):
            nested_dict = True
            break

    if not nested_dict:
        global_dictionary[path] = dictionary
    else:
        for key, value in dictionary.items():
            new_path = f"{path} -> {key}"
            if isinstance(value, dict):
                flatten_dictionary(global_dictionary, value, new_path)
            else:
                global_dictionary[new_path] = value


def is_at_least_one_element_in_list(
        list_of_elements: list,
        search_list: list
) -> bool:
    """
    Checks if at least one element from the list of elements is present in the search list.

    Args:
        list_of_elements (list):
            The list of elements to search.
        search_list (list):
            The list to search in.
    """

    present_in_search_list = [element for element in list_of_elements if element in search_list]
    if len(present_in_search_list) > 0:
        return True
    return False
