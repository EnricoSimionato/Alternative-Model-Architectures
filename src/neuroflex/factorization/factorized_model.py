from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from copy import deepcopy
from enum import Enum
import os
import pickle
from typing import Any, Callable, Optional, override, Union

import numpy as np

import torch
from torch import device

import transformers
from transformers import AutoModel

import imports.peft as peft
from imports.peft import PeftModel

from exporch import Config, get_available_device, Verbose
from exporch.utils import LoggingInterface
from exporch.utils.model_utils import get_parameters
from exporch.utils.parameters_count import count_parameters

from neuroflex.factorization.layers.factorized_embedding_layer import (
    LocalSVDEmbedding,
    GlobalBaseEmbedding,
    GlobalFixedBaseEmbedding
)
from neuroflex.factorization.layers.factorized_layer import (
    MergeableLayer, StructureSpecificGlobalDependent
)
from neuroflex.factorization.layers.factorized_linear_layer import (
    LocalSVDLinear,
    GlobalBaseLinear,
    GlobalFixedBaseLinear,
    GLAMSVDLinear,
    LocalHadamardLinear
)
from neuroflex.utils.plot_utils.heatmap import create_heatmap_global_layers


class FactorizedModel(ABC):
    """
    Factorized model.
    """

    @abstractmethod
    def get_model(
            self
    ) -> None | torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
        """
        Returns the model.

        Returns:
            None | torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                Model.
        """

        return None


class RegularizedTrainingInterface(LoggingInterface, ABC):
    """
    Model with regularization for the training.

    Args:
        initial_regularization_weight (float | torch.Tensor):
            Initial regularization weight.
        max_regularization_weight (float | torch.Tensor):
            Maximum regularization weight.
        start_step_regularization (int):
            Step at which to start the regularization.
        steps_regularization_weight_resets (int):
            Number of steps after which to reset the regularization weights.

    Attributes:
        initial_regularization_weight (torch.Tensor):
            Initial regularization weight.
        fixed_regularization_weight (torch.Tensor):
            Fixed regularization weight.
        adaptive_regularization_weight (torch.Tensor):
            Adaptive regularization weight.
        max_regularization_weight (torch.Tensor):
            Maximum regularization weight.
        start_step_regularization (int):
            Step at which to start the regularization.
        steps_regularization_weight_resets (int):
            Number of steps after which to reset the regularization weight.
        task_loss (torch.Tensor):
            Task loss.
        unweighted_penalization (torch.Tensor):
            Unweighted penalization term.
        weighted_penalization (torch.Tensor):
            Weighted penalization term.
        regularization_loss (torch.Tensor):
            Regularization loss.
    """

    def __init__(
            self,
            initial_regularization_weight: [float, torch.Tensor],
            max_regularization_weight: [float, torch.Tensor],
            start_step_regularization: int,
            steps_regularization_weight_resets: int,
            **kwargs
    ) -> None:
        LoggingInterface.__init__(self)

        self.initial_regularization_weight = initial_regularization_weight
        self.fixed_regularization_weight = None
        self.adaptive_regularization_weight = torch.tensor(initial_regularization_weight)
        self.max_regularization_weight = torch.tensor(max_regularization_weight)
        self.start_step_regularization = start_step_regularization
        self.steps_regularization_weight_resets = steps_regularization_weight_resets

        self.task_loss = 0.0
        self.unweighted_penalization = 0.0
        self.weighted_penalization = 0.0
        self.regularization_loss = 0.0

    def adjust_optimizers_settings(
            self,
            optimizer_settings: list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]],
            **kwargs
    ) -> list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]]:
        """
        Adjusts the optimizer and the learning rate scheduler settings for the training.

        Args:
            optimizer_settings (dict):
                Dictionary containing the optimizer and the learning rate scheduler settings for the training.
            **kwargs:
                Additional keyword arguments.

        Returns:
            list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]]:
                Adjusted optimizer and learning rate scheduler settings for the training.
        """

        return optimizer_settings

    def get_training_penalization_loss(
            self,
            loss: torch.Tensor,
            training_step: int,
            model_device: torch.device,
            **kwargs
    ) -> torch.Tensor:
        """
        Computes the penalization term for the training of the model.

        Args:
            loss (torch.Tensor):
                Current training step loss on the downstream task.
            training_step (int):
                Current training step.
            model_device (torch.device):
                Device where the model is located.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Penalization term.
        """

        if training_step < self.start_step_regularization:
            return torch.tensor(0.0).to(model_device)

        self.regularization_pre_processing(
            training_step,
            **kwargs
        )

        self.task_loss = loss
        self.unweighted_penalization = self.get_unweighted_penalization(**kwargs)
        self.weighted_penalization = self.get_weighted_penalization(
            self.unweighted_penalization,
            loss,
            training_step,
            **kwargs
        )
        self.regularization_loss = self.weighted_penalization * self.adaptive_regularization_weight
        self.regularization_scheduler_step(
            training_step,
            **kwargs
        )

        self.regularization_post_processing(
            training_step,
            **kwargs
        )

        return self.regularization_loss.to(model_device)

    @abstractmethod
    def get_unweighted_penalization(
            self,
            **kwargs
    ) -> torch.Tensor:
        """
        Computes the unweighted penalization term depending on the specific subclass strategy.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Unweighted penalization term.
        """

    def get_weighted_penalization(
            self,
            penalization: torch.Tensor,
            loss: torch.Tensor,
            training_step: int,
            **kwargs
    ) -> torch.Tensor:
        """
        Computes the weighted penalization term.

        Args:
            penalization (torch.Tensor):
                Unweighted penalization term.
            loss (torch.Tensor):
                Current training step loss on the downstream task.
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Weighted penalization term.
        """

        # TODO Weight using the mean of the losses in the previous step

        if self.fixed_regularization_weight is None:
            self.fixed_regularization_weight = torch.tensor(
                (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
        elif (self.steps_regularization_weight_resets > 0 and
              training_step % self.steps_regularization_weight_resets == 0):
            self.fixed_regularization_weight = torch.tensor(
                (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
            self.adaptive_regularization_weight = torch.tensor(
                self.initial_regularization_weight,
                requires_grad=False
            )
            print("Fixed regularization weight reset to", self.fixed_regularization_weight.item(), "and adaptive regularization weight reset to", self.adaptive_regularization_weight.item())
            print("Adaptive regularization weight reset to", self.adaptive_regularization_weight.item())

        return penalization * self.fixed_regularization_weight

    def regularization_scheduler_step(
            self,
            training_step: int,
            **kwargs
    ) -> None:
        """
        Updates the adaptive regularization weight.
        Here the update is linear and the weight goes from the initial regularization weight to the maximum
        regularization weight in steps_regularization_weight_resets steps.

        Args:
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.
        """

        self.adaptive_regularization_weight = torch.tensor(self.initial_regularization_weight).to(self.device) + (
                self.max_regularization_weight - self.initial_regularization_weight
        ) * (
                training_step % self.steps_regularization_weight_resets + 1
        ) / self.steps_regularization_weight_resets

    @torch.no_grad()
    def get_logging_info(
            self,
            **kwargs
    ) -> list:
        """
        Returns additional information to log.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                Additional information to log.
        """

        return [
            {"name": "task_loss", "value": self.task_loss, "on_step": True, "on_epoch": False, "prog_bar": True},
            {"name": "unweighted_penalization", "value": self.unweighted_penalization, "on_step": True, "on_epoch": False, "prog_bar": True},
            {"name": "weighted_penalization", "value": self.weighted_penalization, "on_step": True, "on_epoch": False, "prog_bar": True},
            {"name": "regularization_loss", "value": self.regularization_loss, "on_step": True, "on_epoch": False, "prog_bar": True},
            {"name": "adaptive_regularization_weight", "value": self.adaptive_regularization_weight, "on_step": True, "on_epoch": False, "prog_bar": False},
            {"name": "fixed_regularization_weight", "value": self.fixed_regularization_weight, "on_step": True, "on_epoch": False, "prog_bar": False}
        ]

    def regularization_pre_processing(
            self,
            training_step: int,
            **kwargs
    ) -> None:
        """
        Pre-processes the model after the computation of the regularization term.

        Args:
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.
        """

        pass

    def regularization_post_processing(
            self,
            training_step: int,
            **kwargs
    ) -> None:
        """
        Post-processes the model after the computation of the regularization term.

        Args:
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.
        """

        pass


class GlobalDependentModel(torch.nn.Module, LoggingInterface, ABC):
    """
    Model with global layers replacing some layers of the model.

    Args:
        target_model (PreTrainedModel):
            Pretrained model.
        targets (dict):
            Information to factorize the layers.
            The structure is:
            [
                name_group_1: {
                    attribute_1: ...
                    attribute_2: ...
                    ...
                },
                name_group_2: {
                    attribute_1: ...
                    attribute_2: ...
                    ...
                },
                ...
            ]
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer. If mapping_layer_name_key is not provided and use_names_as_keys is set
            to 'True', then the keys
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by targets or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (Verbose):
            Verbosity level.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        targets (dict):
            Layers to factorize.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers.
        model (PreTrainedModel):
            Pretrained model.
        global_layers (torch.nn.ModuleDict):
            Global layers.
        conversions (dict):
            Mapping of layer types to global-dependent layer classes.
        info (dict):
            Information about the model.
        verbose (Verbose):
            Verbosity level.
    """

    def __init__(
            self,
            target_model: torch.nn.Module = None,
            targets: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        torch.nn.Module.__init__(self)
        LoggingInterface.__init__(self)
        self.verbose = verbose
        self.approximation_stats = None

        if not from_pretrained:
            if target_model is None or targets is None:
                raise ValueError("Both target_model and targets must be provided.")

            self.targets = targets
            if mapping_layer_name_key is None and use_names_as_keys:
                self.mapping_layer_name_key = {layer_name: layer_name for layer_name in targets.keys()}
            else:
                self.mapping_layer_name_key = mapping_layer_name_key

            if preserve_original_model:
                self.model = deepcopy(target_model)
            else:
                self.model = target_model

            self.info = {
                "original_model_parameters": count_parameters(self.model),
                "original_model_trainable_parameters": count_parameters(self.model, only_trainable=True)
            }
            self.global_layers = torch.nn.ModuleDict()
            self.conversions = self.define_conversion(**kwargs)

            self.average_layers = torch.nn.ModuleDict()

            average_matrices = {}
            if remove_average:
                extracted_matrices = {}
                self._collect_matrices_per_name(
                    self.model,
                    extracted_matrices,
                    path="",
                    verbose=self.verbose,
                    **kwargs
                )

                average_matrices = self._compute_average_matrices(
                    extracted_matrices,
                    verbose=self.verbose,
                    **kwargs
                )

            self._processing_before_conversion(**kwargs)
            self._convert_into_global_dependent_model(
                self.model,
                path="",
                average_matrices=average_matrices,
                verbose=verbose,
                **kwargs
            )
            self._processing_after_conversion(**kwargs)

            # Computing the approximation statistics
            #self.approximation_stats = self.compute_approximation_stats()
            #for key in self.approximation_stats:
            #    self.log(f"{key}: {self.approximation_stats[key]}", print_message=True)
            #self.log("", print_message=True)

            # Removing the target_layer attribute from the layers
            extracted_layers = {}
            self._get_wrapped_layers(self.model, extracted_layers)

            for layer in extracted_layers.values():
                layer.delete_target_layer()

            # Computing the number of parameters of the model
            model_parameters = count_parameters(self.model)
            self.info.update({
                "model_parameters": model_parameters,
                "model_trainable_parameters": count_parameters(self.model, only_trainable=True),
                "percentage_parameters": model_parameters / self.info["original_model_parameters"] * 100
            })

            self.log("Information about the factorized model:", print_message=True)
            self.log(f"Number of parameters original model: {self.info['original_model_parameters']}", print_message=True)
            self.log(f"Number of parameters global model: {self.info['model_parameters']}", print_message=True)
            self.log(f"Percentage of parameters: {self.info['percentage_parameters']}%\n", print_message=True)

            self.log("Model converted\n", print_message=True)

    def _get_wrapped_layers(
            self,
            module_tree: [torch.nn.Module | transformers.AutoModel],
            layers_storage: {},
            path: list = None,
    ) -> None:
        """
        Extracts the matrices from the model tree.

        Args:
            module_tree ([torch.nn.Module | transformers.AutoModel]):
                The model tree.
            layers_storage (dict):
                Storage where the extracted layers will be at the end of the extraction.
            path (list, optional):
                The path to the current layer. Defaults to None.
        """

        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            layer_path = copy.deepcopy(path) + [f"{layer_name}"] if path is not None else [f"{layer_name}"]

            if issubclass(type(child), StructureSpecificGlobalDependent):
                layers_storage[tuple(layer_path)] = child
            elif len(child._modules) == 0:
                pass
            else:
                # Recursively calling the function
                self._get_wrapped_layers(module_tree=child, layers_storage=layers_storage, path=layer_path)

    def compute_approximation_stats(
            self
    ) -> dict:
        """
        Computes the approximation statistics.

        Returns:
            dict:
                Approximation statistics.
        """

        wrapped_layers = {}
        self._get_wrapped_layers(self.model, wrapped_layers)
        for layer in wrapped_layers.values():
            layer.compute_approximation_stats()
        concatenated_absolute_approximation_error = torch.tensor([layer.approximation_stats["absolute_approximation_error"] for layer in wrapped_layers.values()])
        concatenated_norm_targets = torch.tensor([layer.approximation_stats["norm_target_layer"] for layer in wrapped_layers.values()])
        concatenated_norm_approximated_layers = torch.tensor([layer.approximation_stats["norm_approximated_layer"] for layer in wrapped_layers.values()])
        mean_relative_approximation_error = torch.mean(torch.tensor([layer.approximation_stats["relative_approximation_error"] for layer in wrapped_layers.values()]))

        return {
            "total_absolute_approximation_error": torch.sum(concatenated_absolute_approximation_error).item(),
            "mean_absolute_approximation_error": torch.mean(concatenated_absolute_approximation_error).item(),
            "mean_relative_approximation_error": mean_relative_approximation_error.item(),
            "sum_norm_targets": torch.sum(concatenated_norm_targets).item(),
            "mean_norm_targets": torch.mean(concatenated_norm_targets).item(),
            "sum_norm_approximated_layers": torch.sum(concatenated_norm_approximated_layers).item(),
            "mean_norm_approximated_layers": torch.mean(concatenated_norm_approximated_layers).item()
        }

    def get_approximation_stats(
            self
    ) -> dict:
        """
        Returns the approximation statistics.

        Returns:
            dict:
                Approximation statistics.
        """

        self.approximation_stats = self.compute_approximation_stats()

        return self.approximation_stats

    def get_model(
            self
    ) -> torch.nn.Module:
        """
        Returns the model.

        Returns:
            torch.nn.Module:
                Model.
        """

        return self.model

    def get_parameters_info(
            self,
            **kwargs
    ) -> dict:
        """
        Returns information about the parameters of the model.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                Information about the parameters of the model.
        """

        return self.info

    def __post_init__(
            self,
            kwargs: dict
    ) -> None:
        """
        Post-initialization method.

        Args:
            kwargs (dict):
                Additional keyword arguments.
        """

        pass

    @abstractmethod
    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the conversion of layers into global-dependent layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-dependent layer classes.
        """

    def change_key(
            self,
            model_tree: torch.nn.Module,
            scope: str,
            previous_key: str,
            new_key: str,
            **kwargs
    ) -> None:
        """
        Changes the key of the global layers.

        Args:
            model_tree (torch.nn.Module):
                Model or module containing layers.
            scope (str):
                Scope of the layer which key has to be changed.
            previous_key (str):
                Previous key of the layer.
            new_key (str):
                New key of the layer.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in model_tree._modules.keys():
            child = model_tree._modules[layer_name]
            if issubclass(type(child), StructureSpecificGlobalDependent):
                child.change_key(
                    scope,
                    previous_key,
                    new_key,
                    **kwargs
                )
            else:
                self.change_key(
                    child,
                    scope,
                    previous_key,
                    new_key,
                    **kwargs
                )

    def _collect_matrices_per_name(
            self,
            model_tree: torch.nn.Module,
            average_matrices: dict,
            path: str = "",
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        """
        Collects the matrices to average per layer name.

        Args:
            model_tree (torch.nn.Module):
                Model or module containing layers.
            average_matrices (dict):
                Dictionary containing the matrices to average per layer name.
            path (str):
                Path to the current layer.
            verbose (Verbose):
                Level of verbosity.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in model_tree._modules.keys():
            child = model_tree._modules[layer_name]
            if len(child._modules) == 0:
                if (type(child) in self.conversions.keys() and
                        layer_name in self.targets.keys()):
                    target_name = layer_name if self.mapping_layer_name_key is None else self.mapping_layer_name_key[layer_name]
                    if target_name not in average_matrices.keys():
                        average_matrices[target_name] = [child]
                    else:
                        average_matrices[target_name].append(child)

            else:
                self._collect_matrices_per_name(
                    child,
                    average_matrices,
                    path + (f"{layer_name}" if path == "" else f".{layer_name}"),
                    verbose=verbose,
                    **kwargs
                )

    def _compute_average_matrices(
            self,
            grouped_layers: dict,
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> dict:
        """
        Computes the average matrices for each given list of matrices, each one identified by a layer name.
        The implementation is written for Linear and Embedding layers.

        Args:
            grouped_layers (dict):
                Dictionary containing the matrices to average per layer name.
            verbose (Verbose):
                Level of verbosity.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                Dictionary containing the average matrices per layer name.
        """

        # Regrouping the layers considering also their dimension
        new_grouped_layers = {}
        for layer_name in grouped_layers.keys():
            if len(grouped_layers[layer_name]) > 1:
                for layer in grouped_layers[layer_name]:
                    shape = layer.weight.shape
                    if f"{layer_name}_({shape[0]},{shape[1]})" not in new_grouped_layers.keys():
                        new_grouped_layers[f"{layer_name}_({shape[0]},{shape[1]})"] = [layer.weight]
                    else:
                        new_grouped_layers[f"{layer_name}_({shape[0]},{shape[1]})"].append(layer.weight)

        average_matrices = {}
        for key in new_grouped_layers.keys():
            if len(new_grouped_layers[key]) > 1:
                average_matrix = torch.mean(
                    torch.stack(
                        new_grouped_layers[key]
                    ),
                    dim=0
                )

                average_matrices[key] = average_matrix

        return average_matrices

    def _processing_before_conversion(
            self,
            **kwargs
    ) -> None:
        pass

    def _convert_into_global_dependent_model(
            self,
            model_tree: torch.nn.Module,
            path: str = "",
            average_matrices: dict = {},
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        """
        Converts layers into global-dependent versions.

        Args:
            model_tree (torch.nn.Module):
                Model or module containing layers.
            path (str):
                Path to the current layer.
            average_matrices (dict):
                Dictionary containing the average matrices per layer name.
            verbose (Verbose):
                Level of verbosity.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in model_tree._modules.keys():
            # Extracting the child from the current module
            child = model_tree._modules[layer_name]
            # If the child has no children, it is a leaf layer and can be eligible for the conversion
            if len(child._modules) == 0:
                # Checking if the layer has to be converted in the current subclass of the GlobalDependentModel.
                # The checks are 3:
                # 1. The layer is in the layers that the model has to convert;
                # 2. The layer is in the targets keys that are the names of the layers to convert;
                # 3. A targets key is contained in some part of the path of the currently considered layer.
                if type(child) in self.conversions.keys():
                    # Additional check to see allow to distinguish layers that have the same name but are in different
                    # paths in the model, e.g. dense layers called 'dense' that are in both the attention component and
                    # in the multi-layer perceptron
                    targets_in_path = [layer_name_ for layer_name_ in self.targets.keys() if layer_name_ in path]

                    if layer_name in self.targets.keys() or len(targets_in_path) > 0:
                        # Initializing the target name with the layer name, if the target name is contained in some part
                        # of the path, we use that label as the target name.
                        target_label = layer_name
                        if len(targets_in_path) > 0:
                            target_label = max(targets_in_path, key=len)
                            if self.verbose > Verbose.INFO:
                                print(f"Among {targets_in_path}, using label: {target_label}")
                        if self.verbose > Verbose.SILENT:
                            print(f"Conversion of {layer_name} in {path} with label {target_label}")

                        # Setting the arguments to pass to the global-dependent layer constructor
                        kwargs_layer = kwargs.copy()
                        kwargs_layer.update(self.targets[target_label])

                        # Setting the average matrix for the layer (if needed)
                        target_name_for_average = self.mapping_layer_name_key[target_label] if self.mapping_layer_name_key is not None else "entire_model_average"
                        average_matrix_key = f"{target_name_for_average}_({child.weight.shape[0]},{child.weight.shape[1]})"
                        kwargs_layer.update(
                            {
                                "average_matrix": None if average_matrix_key not in average_matrices.keys() else average_matrices[average_matrix_key],
                                "path": path
                            }
                        )

                        # Creating the global-dependent layer
                        model_tree._modules[layer_name] = self.conversions[type(child)](
                            child,
                            self.global_layers,
                            self.average_layers,
                            target_name=None if self.mapping_layer_name_key is None else self.mapping_layer_name_key[target_label],
                            **kwargs_layer
                        )

            else:
                # Recursively calling the method on the child, if the child has children
                self._convert_into_global_dependent_model(
                    child,
                    path + (f"{layer_name}" if path == "" else f"_{layer_name}"),
                    average_matrices=average_matrices,
                    verbose=verbose,
                    **kwargs
                )

    def _processing_after_conversion(
            self,
            **kwargs
    ) -> None:
        pass

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            **kwargs
    ) -> None:
        """
        Forward pass of the model.

        Args:
            input_ids:
                Input IDs.
            attention_mask:
                Attention mask.

        Returns:
            Output tensor.
        """

        output = self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        return output

    def generate(
            self,
            inputs: Optional = None,
            generation_config: Optional = None,
            logits_processor: Optional = None,
            stopping_criteria: Optional = None,
            prefix_allowed_tokens_fn: Optional = None,
            synced_gpus: Optional = None,
            assistant_model: Optional = None,
            streamer: Optional = None,
            negative_prompt_ids: Optional = None,
            negative_prompt_attention_mask: Optional = None,
            ** kwargs
    ):
        """
        Generates text.

        Args:
            inputs:
                Inputs.
            generation_config:
                Generation configuration.
            logits_processor:
                Logits processor.
            stopping_criteria:
                Stopping criteria.
            prefix_allowed_tokens_fn:
                Prefix allowed tokens function.
            synced_gpus:
                Synced GPUs.
            assistant_model:
                Assistant model.
            streamer:
                Streamer.
            negative_prompt_ids:
                Negative prompt IDs.
            negative_prompt_attention_mask:
                Negative prompt attention mask.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Generated text.
        """

        return self.model.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs
        )


    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs
    ) -> 'GlobalDependentModel':
        """
        Instantiates a model from a pretrained model file.

        Args:
            pretrained_model_name_or_path (str or os.PathLike):
                Path to the pretrained model file or its name.
            **kwargs:
                Additional keyword arguments.

        Returns:
            GlobalDependentModel:
                The instantiated model.
        """

        global_dependent_model = cls(
            from_pretrained=True,
            **kwargs
        )

        global_dependent_model._load_model(pretrained_model_name_or_path)
        global_dependent_model._load_additional_information(pretrained_model_name_or_path)

        return global_dependent_model

    def _load_model(
            self,
            pretrained_model_path: Union[str, os.PathLike],
    ) -> None:
        """
        Loads the model from the given path.

        Args:
            pretrained_model_path (`str` or `os.PathLike`):
                Directory from which to load.
        """

        self.model = AutoModel.from_pretrained(pretrained_model_path)

    def _load_additional_information(
            self,
            pretrained_model_path: Union[str, os.PathLike],
    ) -> None:
        """
        Loads the additional information of the class.

        Args:
            pretrained_model_path (`str` or `os.PathLike`):
                Directory from which to load.
        """

        with open(os.path.join(pretrained_model_path, "attributes"), "rb") as f:
            self.__dict__.update(pickle.load(f))

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ) -> None:
        """
        Saves the model. The class stores the model using the method from HuggingFace and the other information of the
        class using `pickle`.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        self._save_model(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False, # To change when the model will be changed before storage keeping only once the reference to the global layers
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )

        self._save_additional_information(
            save_directory
        )

    def _save_additional_information(
            self,
            save_directory: Union[str, os.PathLike],
    ) -> None:
        """
        Saves the additional information of the class.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """

        filtered_dict = {key: value for key, value in self.__dict__.items() if key != "model"}

        with open(os.path.join(save_directory, "attributes"), 'wb') as f:
            pickle.dump(filtered_dict, f)

    def _save_model(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ) -> None:
        """
        Saves the model.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        self.model.save_pretrained(
            save_directory,
            is_main_process,
            state_dict,
            save_function,
            push_to_hub,
            max_shard_size,
            safe_serialization,
            variant,
            token,
            save_peft_format,
            **kwargs
        )

    def merge(
            self,
            layers_to_merge: tuple = None,
            **kwargs
    ) -> torch.nn.Module:
        """
        Merges the global layers into the model and returns the result.

        Args:
            layers_to_merge (list):
                List of names of layers to merge.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.nn.Module:
                Model with global layers merged.
        """

        if layers_to_merge is None:
            layers_to_merge = tuple(self.targets.keys())

        merged_model = deepcopy(self.model)
        self._merge_model(merged_model, layers_to_merge, **kwargs)

        return merged_model

    def _merge_model(
            self,
            model_tree: torch.nn.Module,
            layers_to_merge: tuple,
            **kwargs
    ) -> None:
        """
        Utility function to merge the global layers into the model.

        Args:
            model_tree (torch.nn.Module):
                Module containing layers to merge.
            layers_to_merge (list):
                List of names of layers to merge.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in model_tree._modules.keys():
            child = model_tree._modules[layer_name]
            if (isinstance(child, MergeableLayer)) and layer_name in layers_to_merge:
                model_tree._modules[layer_name] = child.merge(**kwargs)
            elif len(child._modules) == 0:
                pass
            else:
                self._merge_model(child, layers_to_merge, **kwargs)

    @property
    def device(self) -> device:
        """
        Device where the model is located.

        Returns:
            Device.
        """

        return next(self.parameters()).device

    def before_training_step(
            self,
            training_step: int,
            **kwargs
    ):
        """
        Method to call before the training step to take some operations.

        Args:
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.
        """

        pass


class LocalSVDModel(GlobalDependentModel):
    """
    Model with LocalSVDLinear layers replacing linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        targets (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by targets or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (int):
            Verbosity level.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model = None,
            targets: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        GlobalDependentModel.__init__(
            self,
            pretrained_model,
            targets,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            remove_average=remove_average,
            from_pretrained=from_pretrained,
            preserve_original_model=preserve_original_model,
            verbose=verbose,
            **kwargs
        )

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {
            torch.nn.Linear: LocalSVDLinear,
            torch.nn.Embedding: LocalSVDEmbedding
        }

        return conversions


class GlobalBaseModel(GlobalDependentModel):
    """
    Model with GlobalBaseLinear layers replacing linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        targets (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by targets or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (int):
            Verbosity level.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model = None,
            targets: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            initialization_type: str = "pseudo-inverse",
            average_svd_initialization: str = "svd_of_average_matrix",
            post_init_train: bool = False,
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        kwargs.update({"initialization_type": initialization_type})
        kwargs.update({"average_svd_initialization": average_svd_initialization})
        GlobalDependentModel.__init__(
            self,
            pretrained_model,
            targets,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            remove_average=remove_average,
            from_pretrained=from_pretrained,
            preserve_original_model=preserve_original_model,
            verbose=verbose,
            **kwargs
        )

        if post_init_train:
            device = get_available_device()

            # Extracting the layers tha have been converted
            global_layers = {}
            self._get_wrapped_layers(self.model, global_layers)

            key_0 = list(global_layers.keys())[0]
            layer_0 = global_layers[key_0]
            targets = {key: layer.get_target_layer() for key, layer in global_layers.items()}
            #targets = {key: torch.ones(layer.get_target_layer().out_features, layer_0.get_target_layer().in_features) for key, layer in global_layers.items()}

            # Configuring the optimizer
            eps = 1e-7 if layer_0.dtype == torch.float16 else 1e-8
            optimizer = torch.optim.AdamW(
                [weight.weight for layer in global_layers.values() for weight in layer.get_local_layers().values()] +
                list(weight.weight for weight in layer_0.get_global_layers().values()),
                lr=1e-4,
                eps=eps
            )

            # Configuring the loss
            loss_fn = torch.nn.MSELoss()

            # Training the matrices to approximate the target matrices
            num_epochs = 1000
            try:
                for epoch in range(num_epochs):
                    # Forward pass
                    key_0 = list(global_layers.keys())[0]
                    loss = loss_fn(global_layers[key_0].weight.to(device), targets[key_0].weight.to(device))
                    for key in list(global_layers.keys())[1:]:
                        loss += loss_fn(global_layers[key].weight.to(device), targets[key].weight.to(device))

                    if epoch % 10 == 0:
                        print("????????????????????????????????????????????????????????????????????????????????????????????")
                        print(f"Epoch {epoch}: {loss}")
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            except (KeyboardInterrupt, SystemExit):
                pass

            print()
            #stats = self.compute_approximation_stats()
            #for key, value in stats.items():
            #    print(f"{key}: {value}")

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Define the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {
            torch.nn.Linear: GlobalBaseLinear,
            torch.nn.Embedding: GlobalBaseEmbedding
        }

        return conversions

    @override
    def _processing_before_conversion(
            self,
            **kwargs
    ) -> None:
        """
        Processing to perform before the conversion of the layers.
        The method computes the global matrices for the layers to be factorized as the average of the SVDs of the layers.

        Args:
            **kwargs:
                Additional keyword arguments.
        """

        if "average_svd_initialization" in kwargs and kwargs["average_svd_initialization"] in ["average_of_svds", "svd_of_average_matrix"]:
            def get_target_layer_name_given_path(target_layer_names: list, path: str) -> str:
                """
                Gets the target layer name given the path of the layer.

                Args:
                    target_layer_names (list):
                        List of target layer names.
                    path (str):
                        Path of the layer.

                Returns:
                    str:
                        Target layer name.
                """

                target_layer_name = None
                for target_name in target_layer_names:
                    if target_name in path:
                        if target_layer_name is None:
                            target_layer_name = target_name
                        else:
                            raise ValueError(
                                f"Multiple target layers found in the path {path}: {target_layer_name} and {target_name}")

                return target_layer_name

            # Obtaining the mapping from the group name in which the layers share the global features to the specific layer
            # name of the layers of the group
            if self.mapping_layer_name_key is None:
                mapping_key_layer_name = {"": [key for key in self.targets.keys()]}
            else:
                mapping_key_layer_name = {}
                for key, value in self.mapping_layer_name_key.items():
                    if value not in mapping_key_layer_name:
                        mapping_key_layer_name[value] = []
                    mapping_key_layer_name[value].append(key)

            for factorization_group, target_names in mapping_key_layer_name.items():
                label = "" if factorization_group == "" else factorization_group + "_"
                # Getting the ranks for the factorization of the layers in the group and checking that they are all equal,
                # otherwise grouping makes no sense
                ranks = {layer_name: self.targets[layer_name]["rank"] for layer_name in target_names}

                # Getting the mapping between the layers path and the actual layers to be used in the computation of the
                # global average layer
                extracted_layers = {}
                get_parameters(self.model, [[layer_name, ] for layer_name in target_names], extracted_layers)

                if len(extracted_layers.keys()) > 0:
                    # Grouping the extracted layers based on the size of the global matrix
                    global_layer_shapes = set([(min(min(layer.in_features, layer.out_features),
                                                    ranks[get_target_layer_name_given_path(target_names, path)]),
                                                layer.in_features) for path, layer in extracted_layers.items()])
                    shape_grouped_extracted_layers = {shape: {} for shape in global_layer_shapes}
                    for path, layer in extracted_layers.items():
                        shape_grouped_extracted_layers[(min(min(layer.in_features, layer.out_features),
                                                            ranks[get_target_layer_name_given_path(target_names, path)]),
                                                        layer.in_features)][path] = layer

                    for shape, shape_group in shape_grouped_extracted_layers.items():
                        print(f"Computing the average global matrix for the group {factorization_group} with shape {shape}...")
                        in_features = shape_group[list(shape_group.keys())[0]].in_features
                        rank = shape[0]

                        if kwargs["average_svd_initialization"] == "average_of_svds":
                            # Computing the SVD on each layer and the average global matrix
                            average_global = None
                            for key, layer in shape_group.items():
                                print(f"Computing SVD for layer {key}...")
                                # Computing the SVD of the layer
                                _, _, vt = np.linalg.svd(layer.weight.data.to(torch.float32).numpy())
                                vt = torch.tensor(vt[:rank, :]).to(layer.weight.dtype)
                                if average_global is None:
                                    average_global = vt
                                else:
                                    average_global += vt

                            # Averaging the sum of the SVDs computed on the layers
                            average_global /= len(list(extracted_layers.keys()))

                        elif kwargs["average_svd_initialization"] == "svd_of_average_matrix":
                            # Computing the avera matrix of the layers
                            average_extrated_layers = None
                            for key, layer in shape_group.items():
                                if average_extrated_layers is None:
                                    average_extrated_layers = layer.weight.data
                                else:
                                    average_extrated_layers += layer.weight.data
                            average_extrated_layers /= len(list(extracted_layers.keys()))

                            # Computing the SVD of the average matrix
                            _, _, vt = np.linalg.svd(average_extrated_layers.to(torch.float32).numpy())
                            average_global = torch.tensor(vt[:rank, :]).to(average_extrated_layers.dtype)

                        else:
                            raise ValueError(f"Average SVD initialization method {kwargs['average_svd_initialization']} not recognized")

                        # Creating the global layer initialized with the average of the SVDs
                        global_layer = torch.nn.Linear(in_features=in_features, out_features=rank, bias=False)
                        with torch.no_grad():
                            global_layer.weight = torch.nn.Parameter(average_global)

                        self.global_layers.add_module(f"{label}({in_features},{rank})", global_layer)


class GlobalFixedBaseModel(GlobalDependentModel):
    """
    Model with GlobalFixedBaseLinear layers replacing the linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        targets (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by targets or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (int):
            Verbosity level.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model = None,
            targets: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            initialization_type: str = "random",
            verbose: int = 0,
            **kwargs
    ) -> None:
        kwargs.update({"initialization_type": initialization_type})
        GlobalDependentModel.__init__(
            self,
            pretrained_model,
            targets,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            remove_average=remove_average,
            from_pretrained=from_pretrained,
            preserve_original_model=preserve_original_model,
            verbose=verbose,
            **kwargs
        )

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {
            torch.nn.Linear: GlobalFixedBaseLinear,
            torch.nn.Embedding: GlobalFixedBaseEmbedding
        }

        return conversions


class LocalHadamardModel(GlobalDependentModel):
    """
    Model with that uses the Hadamard decomposition to compress the model.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        targets (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by targets or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (int):
            Verbosity level.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model=None,
            targets: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: int = 0,
            **kwargs
    ) -> None:
        GlobalDependentModel.__init__(
            self,
            pretrained_model,
            targets,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            remove_average=remove_average,
            from_pretrained=from_pretrained,
            preserve_original_model=preserve_original_model,
            verbose=verbose,
            **kwargs
        )

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {
            torch.nn.Linear: LocalHadamardLinear,
            #torch.nn.Embedding: LocalHadamardEmbedding
        }

        return conversions


class ThresholdingStrategy(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class PruningStrategy(Enum):
    AVERAGE = "average"
    FIRST = "first"
    SECOND = "second"


class GLAMSVDModel(GlobalDependentModel, RegularizedTrainingInterface):
    """
    Model with GLAMSVDLinear layers replacing linear layers and GLAMSVDEmbedding layers
    replacing Embedding layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        targets (dict):
            Layers to factorize. The keys are the names of the layers and the values are dictionaries with at least the
            rank of the decomposition for the layer.
            >> Example:
            >> {
            >>     "layer_name_1": {"rank": 10},
            >>     "layer_name_2": {"rank": 20},
            >> }
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by targets or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (int):
            Verbosity level.
        initial_regularization_weight (float or torch.Tensor):
            Initial weight for regularization.
        max_regularization_weight (float or torch.Tensor):
            Maximum weight for regularization.
        start_step_regularization (int):
            Step at which regularization starts.
        steps_regularization_weight_resets (int):
            Number of steps before the regularization weight resets.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model=None,
            targets: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: int = 0,
            initial_regularization_weight=0.0,
            max_regularization_weight=0.0,
            start_step_regularization=0,
            steps_regularization_weight_resets=0,
            pruning_interval: [float | int] = 0,
            pruning_threshold: float = 0.1,
            thresholding_strategy: ThresholdingStrategy = "averageaverage",
            pruning_strategy: PruningStrategy = "average",
            minimum_number_of_global_layers: int = 1,
            **kwargs
    ) -> None:
        # Ensuring use_names_as_keys is set to True
        use_names_as_keys = True

        GlobalDependentModel.__init__(
            self,
            pretrained_model,
            targets,
            use_names_as_keys=use_names_as_keys,
            mapping_layer_name_key=mapping_layer_name_key,
            remove_average=remove_average,
            from_pretrained=from_pretrained,
            preserve_original_model=preserve_original_model,
            verbose=verbose,
            **kwargs
        )

        RegularizedTrainingInterface.__init__(
            self,
            initial_regularization_weight=initial_regularization_weight,
            max_regularization_weight=max_regularization_weight,
            start_step_regularization=start_step_regularization,
            steps_regularization_weight_resets=steps_regularization_weight_resets,
            **kwargs
        )

        self.pruning_interval = pruning_interval
        self.pruning_threshold = pruning_threshold
        self.thresholding_strategy = ThresholdingStrategy(thresholding_strategy)
        self.pruning_strategy = PruningStrategy(pruning_strategy)
        self.minimum_number_of_global_layers = minimum_number_of_global_layers

        self.label_to_index = None

    def define_conversion(
            self,
            **kwargs
    ) -> dict:
        """
        Defines the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {
            torch.nn.Linear: GLAMSVDLinear,
            #torch.nn.Embedding: GLAMSVDEmbedding
        }

        return conversions

    def adjust_optimizers_settings(
            self,
            optimizer_settings: list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]],
            **kwargs
    ) -> list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]]:
        """
        Adjusts the optimizer and the learning rate scheduler settings for the training.

        Args:
            optimizer_settings (dict):
                Dictionary containing the optimizer and the learning rate scheduler settings for the training.
            **kwargs:
                Additional keyword arguments.

        Returns:
            list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]]:
                Adjusted optimizer and learning rate scheduler settings for the training.
        """

        if len(optimizer_settings) != 2:
            raise ValueError("The model must have two optimizers.")

        optimizers_settings = [
            {
                "optimizer": optimizer_settings[0]["optimizer"],
                "parameters_group": [
                    name
                    for name, _ in self.named_parameters() if 'global_layers' not in name
                ],
                "learning_rate": optimizer_settings[0]["learning_rate"],
                "weight_decay": optimizer_settings[0]["learning_rate"],
                "lr_scheduler": optimizer_settings[0]["lr_scheduler"],
                "warmup_steps": optimizer_settings[0]["warmup_steps"],
                "monitored_metric": optimizer_settings[0]["monitored_metric"]
            },
            {
                "optimizer": "Adam",
                "parameters_group": [
                    name
                    for name, _ in self.named_parameters() if 'global_layers' in name
                ],
                "learning_rate": optimizer_settings[1]["learning_rate"],
                "weight_decay": optimizer_settings[1]["learning_rate"],
                "lr_scheduler": optimizer_settings[1]["lr_scheduler"],
                "warmup_steps": optimizer_settings[1]["warmup_steps"],
                "monitored_metric": optimizer_settings[1]["monitored_metric"]
            }

        ]

        return optimizers_settings

    def get_unweighted_penalization(
            self,
            **kwargs
    ) -> torch.Tensor:
        """
        Returns the penalization term for the training of the global layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Penalization term.
        """

        penalization_term = torch.tensor(0.0, device=self.device)
        l1_norms_table = {}

        for index1, layer1 in enumerate(self.global_layers.values()):
            for index2, layer2 in enumerate(list(self.global_layers.values())[index1+1:]):
                if layer1.weight.shape == layer2.weight.shape:
                    l1_norm = torch.sum(torch.abs((layer1.weight - layer2.weight).flatten()))
                    penalization_term += l1_norm
                    l1_norms_table[(index1, index2)] = l1_norm.item()

        return penalization_term

    def prune_global_layers(
            self,
            **kwargs
    ) -> None:
        """
        Computes the L1-norm of the differences between the global layers and prunes the layers that are similar.

        Args:

            **kwargs:
                Additional keyword arguments.
        """

        print("\n\nChecking if there are global layers to prune...")

        if len(self.global_layers) > max(1, self.minimum_number_of_global_layers):
            layers_keys = list(self.global_layers.keys())
            norm_differences = {}
            for index1, key1 in enumerate(layers_keys):
                layer1 = self.global_layers[key1]
                norm_1 = torch.sum(torch.abs(layer1.weight.flatten()))
                for index2, key2 in enumerate(layers_keys[index1+1:]):
                    layer2 = self.global_layers[key2]
                    norm_2 = torch.sum(torch.abs(layer2.weight.flatten()))
                    if layer1.weight.shape == layer2.weight.shape:
                        norm_difference = torch.sum(torch.abs((layer1.weight - layer2.weight).flatten()))

                        if self.thresholding_strategy == ThresholdingStrategy.ABSOLUTE:
                            # Thresholding based on the absolute difference
                            norm_differences[(key1, key2)] = norm_difference
                        elif self.thresholding_strategy == ThresholdingStrategy.RELATIVE:
                            # Thresholding based on the relative difference
                            norm_differences[(key1, key2)] = norm_difference / torch.sqrt(norm_1) / torch.sqrt(norm_2)
                        else:
                            raise ValueError("Invalid thresholding strategy.")

            min_key = min(norm_differences, key=norm_differences.get)

            if norm_differences[min_key] < self.pruning_threshold:
                print(f"Pruning layers {min_key[0]} and {min_key[1]} with relative norm difference {norm_differences[min_key]}.\n")
                self.apply_pruning(
                    min_key[0],
                    self.global_layers[min_key[0]],
                    min_key[1],
                    self.global_layers[min_key[1]]
                )
            else:
                print(f"No layers to prune, the minimum norm difference is {norm_differences[min_key]} and the threshold is {self.pruning_threshold}.")

            print(f"The number of global layers is {len(self.global_layers)} and the minimum number of global layers is {self.minimum_number_of_global_layers}\n")
        else:
            print(f"No layers to prune, the number of global layers is {len(self.global_layers)}, that is the chosen minimum\n")

    def apply_pruning(
            self,
            key1: str,
            layer1: torch.nn.Module,
            key2: str,
            layer2: torch.nn.Module,
            **kwargs
    ) -> None:
        """
        Sets the global layer of the model.

        Args:
            key1 (str):
                Key of the first layer.
            layer1 (torch.nn.Module):
                First layer.
            key2 (str):
                Key of the second layer.
            layer2 (torch.nn.Module):
                Second layer.
            **kwargs:
                Additional keyword arguments.
        """

        if self.pruning_strategy == PruningStrategy.AVERAGE or self.pruning_strategy == PruningStrategy.FIRST:
            new_key = key1
            previous_key = key2
        elif self.pruning_strategy == PruningStrategy.SECOND:
            new_key = key2
            previous_key = key1
        else:
            raise ValueError("Invalid pruning strategy.")

        self.change_key(
            self.model,
            scope="global",
            previous_key=previous_key,
            new_key=new_key,
            **kwargs
        )

        if self.pruning_strategy == PruningStrategy.AVERAGE:
            with torch.no_grad():
                self.global_layers[key1].weight.data = (layer1.weight + layer2.weight) / 2
            del self.global_layers[key2]
        elif self.pruning_strategy == PruningStrategy.FIRST:
            del self.global_layers[key2]
        elif self.pruning_strategy == PruningStrategy.SECOND:
            del self.global_layers[key1]
        else:
            raise ValueError("Invalid pruning strategy.")

    def before_training_step(
            self,
            training_step: int,
            **kwargs
    ):
        """
        Method to call before the training step to take some operations.

        Args:
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.
        """

        if (self.pruning_interval > 0
                and training_step > 0
                and training_step % self.pruning_interval == 0):
            self.prune_global_layers(
                **kwargs
            )

        if "path_to_storage" not in kwargs.keys():
            raise ValueError("The path to the storage must be provided in order to store the heatmap of the global "
                             "layers usage of the GlamSVD model.")

        self.plot_heatmap_global_layers(
            training_step,
            os.path.join(kwargs["path_to_storage"], "images"),
        )

    def extract_global_keys_list(
            self
    ) -> list:
        """
        Extracts the keys of the global layers for each layer that is using global layers and returns them as a list.

        Returns:
            list:
                List of keys of the global layers usage.
        """

        # Extracting the global keys in dictionary format
        global_keys_dict = {}
        for name, layer in self.model.named_modules():
            for layer_types in self.define_conversion().values():
                if issubclass(type(layer), StructureSpecificGlobalDependent) and issubclass(type(layer), layer_types):
                    global_keys_dict[(layer.layer_index, layer.layer_type)] = layer.get_global_keys()[0]

        # Extracting unique types and indices
        layer_types = sorted(set(layer_type for _, layer_type in global_keys_dict.keys()))
        layer_indices = sorted(set(layer_index for layer_index, _ in global_keys_dict.keys()))

        # Create a dictionary to quickly lookup indices
        type_index_map = {layer_type: index for index, layer_type in enumerate(layer_types)}
        index_index_map = {layer_index: index for index, layer_index in enumerate(layer_indices)}

        # Initialize the 2D list
        num_types = len(layer_types)
        num_indices = len(layer_indices)
        bidimensional_list = [[None] * num_indices for _ in range(num_types)]

        # Populate the 2D list
        for (layer_index, layer_type), value in global_keys_dict.items():
            row = type_index_map[layer_type]
            col = index_index_map[layer_index]
            bidimensional_list[row][col] = value

        return bidimensional_list, layer_types, layer_indices

    def plot_heatmap_global_layers(
            self,
            training_step: int,
            path_to_storage: str,
            **kwargs
    ) -> None:
        """
        Creates the heatmap of the usage of the global layers in the model and appends it to the video of the previous
        heatmaps.
        
        Args:
            training_step (int):
                Current training step.
            path_to_storage (str):
                Path to the storage the heatmaps.
            **kwargs:
                Additional keyword arguments.
        """

        if not os.path.exists(path_to_storage) or not os.path.isdir(path_to_storage):
            raise ValueError("The path to the storage does not exist.")

        global_keys_list, sorted_layer_types, sorted_layer_indexes = self.extract_global_keys_list()

        label_to_index = create_heatmap_global_layers(
            data=global_keys_list,
            title="Global layers used in different parts of the model",
            x_title="Layer indexes",
            y_title="Type of matrix",
            columns_labels=sorted_layer_indexes,
            rows_labels=sorted_layer_types,
            figure_size=(25, 15),
            save_path=path_to_storage,
            heatmap_name=f"heatmap_{training_step}",
            label_to_index=self.label_to_index,
            verbose=self.verbose,
            show=False
        )

        if self.label_to_index is None:
            self.label_to_index = label_to_index


def provide_hyperparameters_for_glam_training(
) -> dict:
    """
    Provides the best hyperparameters for the training of a GLAM model and obtain a model given a budget and some
    features of the training.

    Args:
        None.

    Returns:
        dict:
            Dictionary containing the hyperparameters.
    """

    # TODO write
    pass


class KFCTrainedModel(torch.nn.Module, RegularizedTrainingInterface):
    """
    Model wrapper that allows to penalize the L1-norm of the weights of the pretrained model performing KFC training.

    Args:
        model (torch.nn.Module):
            Model to wrap.
        verbose (int):
            Verbosity level.

    Attributes:
        weights_to_exclude (list):
            List of substrings that identify the weights to exclude from the penalization.
    """

    weights_to_exclude = [
        "lora",
        "vera"
    ]

    def __init__(
            self,
            model: peft.PeftModel,
            initial_regularization_weight: [float, torch.Tensor] = 0.0,
            max_regularization_weight: [float, torch.Tensor] = 0.0,
            start_step_regularization: int = 0,
            steps_regularization_weight_resets: int = 0,
            verbose: int = 0,
            **kwargs
    ) -> None:

        if not issubclass(type(model), PeftModel):
            raise ValueError("The model must be an instance of PeftModel to perform KFC training.")

        torch.nn.Module.__init__(self, **kwargs)
        RegularizedTrainingInterface.__init__(
            self,
            initial_regularization_weight=initial_regularization_weight,
            max_regularization_weight=max_regularization_weight,
            start_step_regularization=start_step_regularization,
            steps_regularization_weight_resets=steps_regularization_weight_resets,
            **kwargs
        )

        self.model = model
        for name, param in self.named_parameters():
            param.requires_grad = True

        self.layers_to_penalize = [
            name
            for name, param in self.named_parameters()
            if not any(substring in name for substring in KFCTrainedModel.weights_to_exclude)
        ]
        self.layers_to_exclude = [
            name
            for name, param in self.named_parameters()
            if any(substring in name for substring in KFCTrainedModel.weights_to_exclude)
        ]
        self.verbose = Verbose(verbose)

    def adjust_optimizers_settings(
            self,
            optimizer_settings: list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]],
            **kwargs
    ) -> list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]]:
        """
        Adjusts the optimizer and the learning rate scheduler settings for the training.

        Args:
            optimizer_settings (dict):
                Dictionary containing the optimizer and the learning rate scheduler settings for the training.
            **kwargs:
                Additional keyword arguments.

        Returns:
            list[dict[str, list[str] | Any] | dict[str, str | list[str] | Any]]:
                Adjusted optimizer and learning rate scheduler settings for the training.
        """

        if len(optimizer_settings) != 2:
            raise ValueError("The model must have two optimizers.")

        optimizers_settings = [
            {
                "optimizer": optimizer_settings[0]["optimizer"],
                "parameters_group": self.layers_to_exclude,
                "learning_rate": optimizer_settings[0]["learning_rate"],
                "weight_decay": optimizer_settings[0]["learning_rate"],
                "lr_scheduler": optimizer_settings[0]["lr_scheduler"],
                "warmup_steps": optimizer_settings[0]["warmup_steps"],
                "monitored_metric": optimizer_settings[0]["monitored_metric"]
            },
            {
                "optimizer": "Adam",
                "parameters_group": self.layers_to_penalize,
                "learning_rate": optimizer_settings[1]["learning_rate"],
                "weight_decay": optimizer_settings[1]["learning_rate"],
                "lr_scheduler": optimizer_settings[1]["lr_scheduler"],
                "warmup_steps": optimizer_settings[1]["warmup_steps"],
                "monitored_metric": optimizer_settings[1]["monitored_metric"]
            }

        ]

        return optimizers_settings

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            **kwargs
    ) -> None:
        """
        Forward pass of the model.

        Args:
            input_ids:
                Input IDs.
            attention_mask:
                Attention mask.

        Returns:
            Output tensor.
        """

        output = self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        return output

    def compute_sum_l1_norms(
            self,
            layers: list
    ) -> torch.Tensor:
        """
        Computes the sum of the L1-norms of the weights of the given layers.

        Args:
            layers (list):
                List of layer names.

        Returns:
            torch.Tensor:
                Sum of the L1-norms of the weights of the given layers.
        """

        sum_l1_norms = torch.tensor(0.0, device=self.device)
        for i, param in enumerate([parameter for name, parameter in self.named_parameters() if name in layers]):
            sum_l1_norms += torch.sum(torch.abs(param.flatten()))

        return sum_l1_norms

    def get_unweighted_penalization(
            self,
            **kwargs
    ) -> torch.Tensor:
        """
        Returns the penalization term that is the L1-norm of the weights of the pretrained model.
        The adapter weights are exlucluded from the computation.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Penalization term.
        """

        return self.compute_sum_l1_norms(self.layers_to_penalize)

    @torch.no_grad()
    def get_logging_info(
            self
    ) -> list:
        """
        Returns additional information to log.

        Returns:
            dict:
                Additional information to log.
        """

        logging_info = super().get_logging_info()
        logging_info.append(
            {"name": "norm of non-regularized weights", "value": self.compute_sum_l1_norms(self.layers_to_exclude), "on_step": True, "on_epoch": False, "prog_bar": False},
        )

        return logging_info

    @property
    def device(self) -> device:
        """
        Device where the model is located.

        Returns:
            Device.
        """

        return next(self.parameters()).device


def update_config_with_model_parameters(
        config: Config,
        model: GlobalDependentModel
) -> None:
    """
    Update the configuration with the number of parameters in the model.

    Args:
        config (Config):
            The configuration to update.
        model (gbm.GlobalDependentModel):
            The model whose parameters are to be counted.
    """

    parameters_info = model.get_parameters_info()
    config.update(parameters_info)
