from __future__ import annotations

import os
import copy
import logging
import pickle as pkl
from typing import Any

import torch

import transformers

from neuroflex.utils.device_utils import get_available_device
from neuroflex.utils.experiment_pipeline.config import Config
from neuroflex.utils.list_utils.list_utils import is_subsequence

from neuroflex.utils.chatbot import load_original_model_for_causal_lm, load_tokenizer_for_causal_lm

import lm_eval

from neuroflex.utils.plotting_utils.heatmap import plot_heatmap

import numpy as np
import re
from collections import OrderedDict


def process_complex_dictionary(
        input_dict: dict[tuple[str, str], dict[str, dict[str, float]]]
) -> tuple:
    # Helper function to extract tuples from string
    def extract_tuples(s):
        return re.findall(r"\('(.*?)', '(.*?)', '(.*?)'\)", s)

    # Initializing OrderedDicts to maintain order and uniqueness
    first_elements = OrderedDict()
    second_elements = OrderedDict()
    tasks = list(next(iter(input_dict.values())).keys())

    # Collecting unique elements and their order
    for key, value in input_dict.items():
        first_tuples = extract_tuples(key[0])
        second_tuples = extract_tuples(key[1])

        for t in first_tuples:
            if t[0] not in first_elements:
                first_elements[t[0]] = len(first_elements)
        for t in second_tuples:
            if t[0] not in second_elements:
                second_elements[t[0]] = len(second_elements)

    # Initialize performance arrays
    performance_arrays = {task: np.full((len(first_elements), len(second_elements)), np.nan) for task in tasks}

    # Second pass: fill in the performance arrays
    for key, value in input_dict.items():
        first_tuples = extract_tuples(key[0])
        second_tuples = extract_tuples(key[1])

        for first in first_tuples:
            for second in second_tuples:
                row = first_elements[first[0]]
                col = second_elements[second[0]]
                for task in tasks:
                    performance_arrays[task][row, col] = value[task]['metric']

    return list(first_elements.keys()), list(second_elements.keys()), performance_arrays


class LayerSwitchingWrapperModel:
    """
    Class to switch layers in a model.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The layers to switch.

    Attributes:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The layers to switch.
    """

    def __init__(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel],
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple] = None,
    ) -> None:

        self.model = model
        self.destination_layer_path_source_layer_path_mapping = destination_layer_path_source_layer_path_mapping
        self.overwritten_layers = {}

        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.switch_layers()

    def get_model(
            self
    ) -> [transformers.PreTrainedModel | transformers.AutoModel]:
        """
        Returns the model.

        Returns:
            [transformers.PreTrainedModel | transformers.AutoModel]:
                The model.
        """

        return self.model

    def get_destination_layer_path_source_layer_path_mapping(
            self
    ) -> dict[list | tuple: list | tuple]:
        """
        Returns the layers to switch.

        Returns:
            dict[list | tuple: list | tuple]:
                The layers to switch.
        """

        return self.destination_layer_path_source_layer_path_mapping

    def set_model(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel]
    ) -> None:
        """
        Sets the model.

        Args:
            model ([transformers.PreTrainedModel | transformers.AutoModel]):
                The model.
        """

        self.model = model
        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.switch_layers()

    def set_destination_layer_path_source_layer_path_mapping(
            self,
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple]
    ) -> None:
        """
        Sets the layers to switch.

        Args:
            destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
                The layers to switch.
        """

        self.destination_layer_path_source_layer_path_mapping = destination_layer_path_source_layer_path_mapping
        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.switch_layers()

    def switch_layers(
            self
    ) -> None:
        """
        Switches the layers of the model.
        """

        source_paths = set(self.get_destination_layer_path_source_layer_path_mapping().values())
        source_layer_path_source_layer_mapping = {source_path: None for source_path in source_paths}
        self._extract_source_layers(self.get_model(), source_layer_path_source_layer_mapping)
        if any([source_layer_path_source_layer_mapping[source_path] is None
                for source_path in source_layer_path_source_layer_mapping.keys()]):
            raise Exception("Some layers could not be extracted.")
        destination_layer_path_source_layer_mapping = {
            destination_path: source_layer_path_source_layer_mapping[source_path]
            for destination_path, source_path in self.get_destination_layer_path_source_layer_path_mapping().items()
        }
        self._fill_destination_layers(self.model, destination_layer_path_source_layer_mapping)

    def _extract_source_layers(
            self,
            module_tree: [transformers.PreTrainedModel | transformers.AutoModel | torch.Module],
            source_layer_path_source_layer_mapping: dict[list | tuple: list | tuple],
            path: list = None
    ) -> None:
        """
        Extracts the source layers from the model.

        Args:
            module_tree ([transformers.PreTrainedModel | transformers.AutoModel | torch.Module]):
                The module tree.
            source_layer_path_source_layer_mapping (dict[str: str]):
                The destination layer path source layer mapping.
            path (list, optional):
                The path. Defaults to None.
        """

        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            source_paths = list(source_layer_path_source_layer_mapping.keys())
            source_paths_in_current_path = [is_subsequence(source_path, path + [f"{layer_name}"] if path is not None else [f"{layer_name}"]) for source_path in source_paths]
            if sum(source_paths_in_current_path) > 1:
                raise Exception("Multiple layers have the same path.")
            if any(source_paths_in_current_path):
                # Storing the child in the destination layer path source layer mapping
                source_path = source_paths[source_paths_in_current_path.index(True)]
                source_layer_path_source_layer_mapping[source_path] = child
            elif len(child._modules) == 0:
                # If the child has no children, we reached a leaf node and we do nothing
                pass
            else:
                # Recursively calling the method on the child, if the child has children
                new_path = copy.copy(path) + [f"{layer_name}"] if path != None else [f"{layer_name}"]
                self._extract_source_layers(
                    child,
                    source_layer_path_source_layer_mapping,
                    new_path
                )

    def _fill_destination_layers(
            self,
            module_tree: [transformers.PreTrainedModel | transformers.AutoModel | torch.Module],
            destination_layer_path_source_layer_mapping: dict[list | tuple: torch.Module],
            path: list = None
    ) -> None:
        """
        Fills the destination layers with the source layers.

        Args:
            module_tree ([transformers.PreTrainedModel | transformers.AutoModel | torch.Module]):
                The module tree.
            destination_layer_path_source_layer_mapping (dict[str: torch.Module]):
                The destination layer path source layer mapping.
            path (list, optional):
                The path. Defaults to None.
        """

        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            destination_paths = list(destination_layer_path_source_layer_mapping.keys())
            destination_paths_in_current_path = [is_subsequence(destination_path, path + [f"{layer_name}"] if path is not None else [f"{layer_name}"]) for destination_path in destination_paths]
            if sum(destination_paths_in_current_path) > 1:
                raise Exception("Multiple layers have the same path.")
            if any(destination_paths_in_current_path):
                # Setting the new layer from the source path in the destination layer
                destination_path = destination_paths[destination_paths_in_current_path.index(True)]
                ########################################################################################################
                # Storing the overwritten layer in order be able to reset the switch.
                # For future changes: the very same method is used to reset the switch, passing a COPY of the dictionary
                # overwritten_layers, using the very same instance the reset does not work.
                self.overwritten_layers[destination_path] = module_tree._modules[layer_name]
                ########################################################################################################
                module_tree._modules[layer_name] = destination_layer_path_source_layer_mapping[destination_path]
            elif len(child._modules) == 0:
                # If the child has no children, we reached a leaf node and we do nothing
                pass
            else:
                # Recursively calling the method on the child, if the child has children
                new_path = copy.copy(path) + [f"{layer_name}"] if path != None else [f"{layer_name}"]
                self._fill_destination_layers(
                    child,
                    destination_layer_path_source_layer_mapping,
                    new_path
                )

    def reset_switch(
            self
    ) -> None:
        """
        Resets the switch.
        """

        if len(self.overwritten_layers) == 0:
            raise Exception("The layers have not been switched.")

        self._fill_destination_layers(self.model, copy.copy(self.overwritten_layers))
        self.overwritten_layers = {}

    def __str__(self):
        """
        Returns the string representation of the object.
        """

        return self.model.__str__()


lm_eval_task_args = {
    "mmlu": {"subset": "high_school_mathematics"},
    "hellaswag": {}
}


def perform_layer_redundancy_analysis_launcher(
        config: Config,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_layer_redundancy_analysis_launcher in redundancy_hunter.py.")

    # Getting the parameters related to the paths from the configuration
    file_available, file_path, directory_path = [
        config.get(name)
        for name in ["file_available", "file_path", "directory_path"]
    ]

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
    else:
        data = None

    perform_layer_redundancy_analysis(config, data)


def perform_layer_redundancy_analysis(
        config: Config,
        data: Any = None,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_layer_redundancy_analysis in redundancy_hunter.py.")

    if data is None:
        # Getting the parameters from the configuration
        device = get_available_device(config.get("device") if config.contains("device") else None, just_string=True)
        batch_size = config.get("batch_size") if config.contains("batch_size") else 4

        performance_dict = {}

        # Loading the model and the tokenizer
        base_model = load_original_model_for_causal_lm(config)
        logger.info(f"Model loaded.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = load_tokenizer_for_causal_lm(config)
        logger.info(f"Tokenizer loaded.")

        # Setting the parameters for the layer switching
        targets_lists = config.get("targets")
        num_layers = config.get("num_layers")
        destination_layer_path_source_layer_path_mapping_list = [
            {
                tuple(el if el != "layer_index" else f"{i}" for el in targets):
                    tuple(el if el != "layer_index" else f"{j}" for el in targets)
                for targets in targets_lists
            }
            for i in range(num_layers)
            for j in range(num_layers) if (i != j or (i == 0 and j== 0))
        ]

        # Wrapping the model to move the layers
        model_wrapper = LayerSwitchingWrapperModel(base_model, None)
        logger.info(f"Model wrapped.")

        for destination_layer_path_source_layer_path_mapping in destination_layer_path_source_layer_path_mapping_list:
            logger.info(f"Evaluating the variance destination_layer_path_source_layer_path_mapping: {destination_layer_path_source_layer_path_mapping}")
            print(f"Evaluating the variance destination_layer_path_source_layer_path_mapping: {destination_layer_path_source_layer_path_mapping}")

            model_wrapper.set_destination_layer_path_source_layer_path_mapping(destination_layer_path_source_layer_path_mapping)
            logger.info(f"Layers switched.")

            # Defining the evaluation parameters
            evaluation_args = {
                "num_fewshot": 5,
                "batch_size": batch_size,
                "device": device
            }

            logger.info(f"Starting the evaluation of the model.")
            # Evaluating the model

            results = lm_eval.simple_evaluate(
                model="hf",
                model_args={"pretrained": model_wrapper.get_model().to(get_available_device(device)), "tokenizer": tokenizer, "backend": "causal"},
                tasks=config.get("dataset_id"),
                batch_size=evaluation_args["batch_size"],
                device=evaluation_args["device"]
            )
            logger.info(f"Model evaluated.")
            filtered_results = results["results"]
            logger.info(f"Results: {filtered_results}")

            #filtered_results = {"task": {"metric": 0.0}}

            performance_dict[(str(destination_layer_path_source_layer_path_mapping.keys()), str(destination_layer_path_source_layer_path_mapping.values()))] = filtered_results
            logger.info(f"Performance dictionary updated with the results.")

            model_wrapper.reset_switch()
            logger.info(f"Layers reset.")

        data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)
        logger.info("Trying to store the data.")
        # Saving the data
        with open(config.get("file_path"), "wb") as f:
            pkl.dump(data, f)
        logger.info("Data stored.")

    destination_layer_path_source_layer_path_mapping_list, destination_layer_path_source_layer_path_mapping_list = data

    #row_labels = [str(destination_layer_path_source_layer_path_mapping) for destination_layer_path_source_layer_path_mapping in destination_layer_path_source_layer_path_mapping_list]
    #column_labels = []

    """
    for task in config.get("dataset_id"):
        results_matrix = []
        logger.info(f"Printing the results for task: {task}")
        destination_layer_path_source_layer_path_mapping_list
        plot_heatmap(
            [[]]
       )
    """


