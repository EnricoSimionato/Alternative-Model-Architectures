import os
import pickle
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Optional, Callable, Union

import torch
import torch.nn as nn
from torch import device

from gbm.layers.global_dependent_layer import (
    MergeableLayer
)
from gbm.layers.gdl_linear import (
    LocalSVDLinear,
    GlobalBaseLinear,
    GlobalFixedBaseLinear
)

from gbm.layers.gdl_embedding import (
    LocalSVDEmbedding,
    GlobalBaseEmbedding,
    GlobalFixedBaseEmbedding
)

from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

from gbm.utils import count_parameters


class GlobalDependentModel(ABC, nn.Module):
    """
    Model with global layers replacing some layers of the model.

    Args:
        target_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
            Layers to factorize.
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by target_layers or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (bool):
            Whether to print information about the conversion.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        target_layers (dict):
            Layers to factorize.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers.
        model (PreTrainedModel):
            Pretrained model.
        global_layers (nn.ModuleDict):
            Global layers.
        conversions (dict):
            Mapping of layer types to global-dependent layer classes.
        info (dict):
            Information about the model.
    """

    def __init__(
            self,
            target_model: nn.Module = None,
            target_layers: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: bool = False,
            **kwargs
    ) -> None:
        super(GlobalDependentModel, self).__init__()

        if not from_pretrained:
            if target_model is None or target_layers is None:
                raise ValueError("Both target_model or target_layers must be provided.")

            self.target_layers = target_layers
            if mapping_layer_name_key is None and use_names_as_keys:
                self.mapping_layer_name_key = {layer_name: layer_name for layer_name in target_layers.keys()}
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
            self.global_layers = nn.ModuleDict()
            self.conversions = self.define_conversion(**kwargs)

            self.average_layers = nn.ModuleDict()

            average_matrices = {}
            if remove_average:
                extracted_matrices = {}
                self._collect_matrices_per_name(
                    self.model,
                    extracted_matrices,
                    path="",
                    verbose=verbose,
                    **kwargs
                )

                average_matrices = self._compute_average_matrices(
                    extracted_matrices,
                    verbose=verbose,
                    **kwargs
                )

            self._convert_into_global_dependent_model(
                self.model,
                path="",
                average_matrices=average_matrices,
                verbose=verbose,
                **kwargs
            )
            self.__post_init__(kwargs)

            model_parameters = count_parameters(self.model)
            self.info.update(
                {
                    "model_parameters": model_parameters,
                    "model_trainable_parameters": count_parameters(self.model, only_trainable=True),
                    "percentage_parameters": model_parameters / self.info["original_model_parameters"] * 100
                }
            )

            if verbose:
                print(f"Number of parameters original model: {self.info['original_model_parameters']}")
                print(f"Number of parameters global model: {self.info['model_parameters']}")
                print(f"Percentage of parameters: {self.info['percentage_parameters']}%")
                print()
                print("Model converted")

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

    def _collect_matrices_per_name(
            self,
            model_tree: nn.Module,
            average_matrices: dict,
            path: str = "",
            verbose: bool = False,
            **kwargs
    ) -> None:
        """
        Collects the matrices to average per layer name.

        Args:
            model_tree (nn.Module):
                Model or module containing layers.
            average_matrices (dict):
                Dictionary containing the matrices to average per layer name.
            path (str):
                Path to the current layer.
            verbose (bool):
                Whether to print information about the conversion.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in model_tree._modules.keys():
            child = model_tree._modules[layer_name]
            if len(child._modules) == 0:
                if (type(child) in self.conversions.keys() and
                        layer_name in self.target_layers.keys()):
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
            verbose: bool = False,
            **kwargs
    ) -> dict:
        """
        Computes the average matrices for each given list of matrices, each one identified by a layer name.
        The implementation is written for Linear and Embedding layers.

        Args:
            grouped_layers (dict):
                Dictionary containing the matrices to average per layer name.
            verbose (bool):
                Whether to print information about the conversion.
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

    def _convert_into_global_dependent_model(
            self,
            model_tree: nn.Module,
            path: str = "",
            average_matrices: dict = {},
            verbose: bool = False,
            **kwargs
    ) -> None:
        """
        Converts layers into global-dependent versions.

        Args:
            model_tree (nn.Module):
                Model or module containing layers.
            path (str):
                Path to the current layer.
            average_matrices (dict):
                Dictionary containing the average matrices per layer name.
            verbose (bool):
                Whether to print information about the conversion.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in model_tree._modules.keys():
            child = model_tree._modules[layer_name]
            if len(child._modules) == 0:
                if (type(child) in self.conversions.keys() and
                        layer_name in self.target_layers.keys()):
                    if verbose:
                        print(f"Conversion of {layer_name} in {path}")
                    kwargs_layer = kwargs.copy()
                    kwargs_layer.update(self.target_layers[layer_name])
                    average_matrix_key = f"{self.mapping_layer_name_key[layer_name]}_({child.weight.shape[0]},{child.weight.shape[1]})"
                    kwargs_layer.update(
                        {
                            "average_matrix": None if average_matrix_key not in average_matrices.keys() else average_matrices[average_matrix_key]
                        }
                    )

                    model_tree._modules[layer_name] = self.conversions[type(child)](
                        child,
                        self.global_layers,
                        self.average_layers,
                        target_name=None if self.mapping_layer_name_key is None else self.mapping_layer_name_key[layer_name],
                        **kwargs_layer
                    )

            else:
                self._convert_into_global_dependent_model(
                    child,
                    path + (f"{layer_name}" if path == "" else f".{layer_name}"),
                    average_matrices=average_matrices,
                    verbose=verbose,
                    **kwargs
                )

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
    ) -> nn.Module:
        """
        Merges the global layers into the model and returns the result.

        Args:
            layers_to_merge (list):
                List of names of layers to merge.
            **kwargs:
                Additional keyword arguments.

        Returns:
            nn.Module:
                Model with global layers merged.
        """

        if layers_to_merge is None:
            layers_to_merge = tuple(self.target_layers.keys())

        merged_model = deepcopy(self.model)
        self._merge_model(merged_model, layers_to_merge, **kwargs)

        return merged_model

    def _merge_model(
            self,
            model_tree: nn.Module,
            layers_to_merge: tuple,
            **kwargs
    ) -> None:
        """
        Utility function to merge the global layers into the model.

        Args:
            model_tree (nn.Module):
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


class LocalSVDModel(GlobalDependentModel):
    """
    Model with LocalSVDLinear layers replacing linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
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
            grouping imposed by target_layers or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (bool):
            Whether to print information about the conversion.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model = None,
            target_layers: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: bool = False,
            **kwargs
    ) -> None:
        super(LocalSVDModel, self).__init__(
            pretrained_model,
            target_layers,
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
            nn.Linear: LocalSVDLinear,
            nn.Embedding: LocalSVDEmbedding
        }

        return conversions


class GlobalBaseModel(GlobalDependentModel):
    """
    Model with GlobalBaseLinear layers replacing linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
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
            grouping imposed by target_layers or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (bool):
            Whether to print information about the conversion.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model = None,
            target_layers: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: bool = False,
            **kwargs
    ) -> None:
        super(GlobalBaseModel, self).__init__(
            pretrained_model,
            target_layers,
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
        Define the conversion of layers into global-base layers.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            Dictionary mapping layer types to corresponding global-base layer classes.
        """

        conversions = {
            nn.Linear: GlobalBaseLinear,
            nn.Embedding: GlobalBaseEmbedding
        }

        return conversions


class GlobalFixedBaseModel(GlobalDependentModel):
    """
    Model with GlobalFixedBaseLinear layers replacing the linear layers.

    Args:
        pretrained_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
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
            grouping imposed by target_layers or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (bool):
            Whether to print information about the conversion.
        **kwargs:
            Additional keyword arguments.
    """

    def __init__(
            self,
            pretrained_model = None,
            target_layers: dict = None,
            use_names_as_keys: bool = False,
            mapping_layer_name_key: dict = None,
            remove_average: bool = False,
            from_pretrained: bool = False,
            preserve_original_model: bool = False,
            verbose: bool = False,
            **kwargs
    ) -> None:
        super(GlobalFixedBaseModel, self).__init__(
            pretrained_model,
            target_layers,
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
            nn.Linear: GlobalFixedBaseLinear,
            nn.Embedding: GlobalFixedBaseEmbedding
        }

        return conversions
