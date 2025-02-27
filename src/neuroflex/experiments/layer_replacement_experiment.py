import copy
import gc
from typing import override, Union

import torch

import transformers

from peft import PeftMixedModel, PeftModel

from exporch import Config

from exporch.utils.model_utils import get_parameters
from neuroflex.experiments.fine_tuning_experiment import FineTuningExperiment
from neuroflex.pruning.replacement_model import get_layer_replaced_model
from neuroflex.utils.adapters_utils.adapters_utils import get_adapted_model


class LayerReplacementFineTuningExperiment(FineTuningExperiment):
    """
    Class to perform the evaluation of a model with unique average layer on some benchmarks, the fine-tuning of the
    layer of the model that has been replaced and the evaluation on the same benchmarks following the re-training.
    """

    mandatory_keys = ["replacement_methods", "num_layers", "target_layers"]
    deepcopy = False

    @override
    def prepare_models(
            self,
            base_model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer
    ) -> dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None]:
        """
        Returns the prepared models to be evaluated.

        Args:
            base_model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None):
                The original model to be prepared.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer of the model.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None]:
                The prepared models to be evaluated.
        """

        self.log(f"Preparing the models using replacement methods {self.config.get('replacement_methods')}.")
        prepared_models = {}
        try:
            for replacement_method in self.config.get("replacement_methods"):
                loaded_model = self.load(f"{replacement_method}.pt", "pt")
                if loaded_model is not None:
                    prepared_models[f"{replacement_method}"] = loaded_model
                    self.log(f"Model with replaced layers loaded from storage.",print_message=True)
                else:
                    prepared_models[f"{replacement_method}"] = get_layer_replaced_model(
                        copy.deepcopy(base_model), replacement_method, self.get_layers_replacement_mapping(), self.deepcopy, self.config
                    )
                    self.log(f"Model prepared replacing the layers.", print_message=True)
                self.log(f"Replacement method: {replacement_method}.", print_message=True)
                self.store(prepared_models[f"{replacement_method}"], f"{replacement_method}.pt", "pt")
                if self.is_low_memory_mode():
                    del prepared_models[f"{replacement_method}"]
                    gc.collect()
                    prepared_models[f"{replacement_method}"] = None
                    self.log("Model deleted from memory.", print_message=True)
        except Exception as e:
            self.log(f"Error preparing the model: {e}")
            raise e

        self.log(f"Models prepared: {prepared_models.keys()}")

        return prepared_models

    def get_layers_replacement_mapping(
            self
    ) -> dict[tuple, tuple]:
        """
        Returns the mapping to be used to modify the model.
        The mapping is a dictionary where the keys are the paths to the layers to be replaced, and the values are the
        paths to the layers that will replace them. The meaning of the paths is defined by the replacement method.
        By default, the dictionary maps the layers of the model to themselves.

        Returns:
            dict[tuple, tuple]:
                The mapping to be used to modify the model.
        """

        self.log(f"Creating the layers replacement mapping.")

        targets_lists = self.config.get("target_layers")
        num_layers = self.config.get("num_layers")
        excluded_blocks = self.config.get("excluded_blocks") if self.config.contains("excluded_blocks") else []

        return {
            tuple(el if el != "block_index" else f"{i}" for el in targets):
                tuple(el if el != "block_index" else f"{i}" for el in targets) for targets in targets_lists
                for i in range(num_layers) if i not in excluded_blocks
        }

    def get_layers_to_train(
            self,
            model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel
    ) -> dict:
        """
        Returns the layers to train.

        Args:
            model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The model to fine-tune.

        Returns:
            dict:
                The layers to train.
        """

        excluded_blocks = self.config.get("excluded_blocks") if self.config.contains("excluded_blocks") else []

        layers_to_train = super().get_layers_to_train(model)
        layers_to_train_without_excluded_blocks = {}
        for path, layer in layers_to_train.items():
            if all(str(el) != str(excluded_block) for el in path for excluded_block in excluded_blocks):
                layers_to_train_without_excluded_blocks[path] = layer

        return layers_to_train_without_excluded_blocks


class LayerReplacementFineTuningEntireModelExperiment(LayerReplacementFineTuningExperiment):
    """
    Class to perform the evaluation of a model with unique average layer on some benchmarks, the fine-tuning of the
    entire model and the evaluation on the same benchmarks following the re-training.
    """

    @override
    def get_layers_to_train(
            self,
            model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel
    ) -> dict:
        """
        Prepares the fine-tuning of the models. This method can be overridden to add more operations.

        Args:
            model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The model to fine-tune.

        Returns:
            list:
                The layers to train.
        """

        not_trainable_names = ["norm", "layernorm", "ln", "batchnorm", "bn", "embedding", "emb", "position", "pos", "activation", "act"]
        target_names = []
        for name, parameter in model.named_parameters():
            if not any(el in name.lower() for el in not_trainable_names):
                layer_name = name.split(".")[:-1]
                if layer_name not in target_names:
                    target_names.append(layer_name)

        extracted_layers = {}
        get_parameters(model, target_names, extracted_layers, blacklist=self.config.get("blacklist") if self.config.contains("blacklist") else [])

        layers_to_train = {}
        for path, layer in extracted_layers.items():
            if isinstance(layer, torch.nn.Embedding) or any(el in path for el in ["embedding", "emb"]):
                self.log(f"The layer with path {path} is an embedding layer so it is set to be not trainable.")
            elif isinstance(layer, torch.nn.LayerNorm) or any(el in path for el in ["norm", "layernorm", "ln", "batchnorm", "bn"]):
                self.log(f"The layer with path {path} is a normalization layer so it is set to be not trainable.")
            else:
                layers_to_train[tuple(path)] = layer

        return layers_to_train


class LayerReplacementFineTuningAdapterOnTargetsExperiment(LayerReplacementFineTuningExperiment):
    """
    Class to perform the evaluation of a model with unique average layer on some benchmarks, the fine-tuning of the
    replaced layers of the model using adapters on the aggregated layers and the evaluation on the same benchmarks
    following the re-training.
    """

    mandatory_keys = ["adapter_method", "adapted_layers"]

    @override
    def prepare_fine_tuning(
            self,
            prepared_model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None
    ) -> torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None:
        """
        Prepares the model for fine-tuning by setting the layers to train.
        This method can be overridden to do different operations.

        Args:
            prepared_model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None):
                The model to fine-tune.

        Returns:
            torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None:
                The model prepared for fine-tuning.
        """

        # Wrapping the model with the adapter
        adapted_model = self._get_adapted_model(prepared_model)
        self.log(f"Model with adapters: {adapted_model}.", print_message=True)

        return adapted_model

    def _get_adapted_model(
            self,
            model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel
    ) -> Union[PeftModel, PeftMixedModel]:
        """
        Returns the adapted model to be trained given the model to be wrapped by the adapter.

        Args:
            model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The model to be adapted.

        Returns:
            Union[PeftModel, PeftMixedModel]:
                The adapted model.
        """

        config_dict = self.config.get_dict(["adapter_method", "adapted_layers"])
        config_dict.update({
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        })
        config_dict.update(self.config.get_dict(["lora_dropout", "bias", "task_type"]))

        self.log(f"Creating the adapted model with the following configuration: {config_dict}.")
        try:
            adapted_model = get_adapted_model(model.model, Config.convert_to_config(config_dict))
            model.model = adapted_model
            return model
        except Exception as e:
            raise e


class LayerReplacementFineTuningDifferentAdapterOnTargetsExperiment(LayerReplacementFineTuningAdapterOnTargetsExperiment):
    """
    Class to perform the evaluation of a model with unique average layer on some benchmarks, the fine-tuning of the
    replaced layers of the model using different adapters on the aggregated layers and the evaluation on the same
    benchmarks following the re-training.
    """

    mandatory_keys = ["adapter_method", "adapted_layers"]
    deepcopy = True

    def _get_adapted_model(
            self,
            model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel
    ) -> Union[PeftModel, PeftMixedModel]:
        """
        Returns the adapted model to be trained given the model to be wrapped by the adapter.

        Args:
            model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The model to be adapted.

        Returns:
            Union[PeftModel, PeftMixedModel]:
                The adapted model.
        """

        config_dict = self.config.get_dict(["adapter_method", "adapted_layers"])
        config_dict.update({
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        })
        config_dict.update(self.config.get_dict(["lora_dropout", "bias", "task_type"]))

        self.log(f"Creating the adapted model with the following configuration: {config_dict}.")
        try:
            adapted_model = get_adapted_model(model.model, Config.convert_to_config(config_dict))
            print(adapted_model)
            #base_layer = adapted_model.base_model.model.bert.encoder.layer[0].attention.self.key.base_layer
            base_layer = adapted_model.base_model.model.model.layers[0].mlp.gate_proj.base_layer.weight
            #base_layer = adapted_model.base_model.model.model.layers[0].self_attn.q_proj.base_layer
            for i in range(1, self.config.get("num_layers")):
                print(f"Layer {i}")
                #adapted_model.bert.encoder.layer[i].attention.self.key.base_layer = base_layer

                #del adapted_model.base_model.model.model.layers[i].self_attn.q_proj.base_layer
                #del adapted_model.base_model.model.model.layers[i].mlp.gate_proj.base_layer

                #adapted_model.base_model.model.model.layers[i].self_attn.q_proj.base_layer = base_layer
                adapted_model.base_model.model.model.layers[i].mlp.gate_proj.base_layer.weight = base_layer
                torch.cuda.empty_cache()
                
            for name, param in adapted_model.named_parameters():
                print(name, param)
            model.model = adapted_model
            print(adapted_model)
            return model
        except Exception as e:
            raise e