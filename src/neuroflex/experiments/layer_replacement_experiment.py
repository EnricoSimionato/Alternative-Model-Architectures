import copy
from typing import override, Union

import torch

import transformers

from peft import PeftModel, PeftMixedModel

from exporch import Config
from neuroflex.experiments.fine_tuning_experiment import FineTuningExperiment
from neuroflex.pruning.replacement_model import get_layer_replaced_model
from neuroflex.experiments.extratomove import get_parameters
from neuroflex.utils.adapters_utils.adapters_utils import get_adapted_model


class LayerReplacementFineTuningExperiment(FineTuningExperiment):
    """
    Class to perform the evaluation of a model with with some layers replaced following the given strategy on some
    benchmarks, the fine-tuning of the model and the evaluation on the same benchmarks again.
    """

    mandatory_keys = ["replacement_methods", "num_layers", "targets"]

    @override
    def prepare_models(
            self,
            base_model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer
    ) -> dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
        """
        Returns the prepared models to be evaluated.

        Args:
            base_model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The original model to be prepared.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer of the model.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
                The prepared models to be evaluated.
        """

        self.log(f"Preparing the models using replacement methods {self.config.get('replacement_methods')}.")

        prepared_models = {}

        base_model.cpu()
        try:
            for replacement_method in self.config.get("replacement_methods"):
                loaded_model = self.load(f"{replacement_method}.pt", "pt")
                if loaded_model is None:
                    prepared_models[f"{replacement_method}"] = get_layer_replaced_model(
                        #copy.deepcopy(base_model), replacement_method, self.get_layers_replacement_mapping(), self.config
                        base_model, replacement_method, self.get_layers_replacement_mapping(), self.config
                    ).get_model()
                else:
                    prepared_models[f"{replacement_method}"] = loaded_model

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

        targets_lists = self.config.get("targets")
        num_layers = self.config.get("num_layers")

        return {
            tuple(el if el != "block_index" else f"{i}" for el in targets):
                tuple(el if el != "block_index" else f"{i}" for el in targets) for targets in targets_lists
                for i in range(num_layers)
        }


class LayerReplacementFineTuningEntireModelExperiment(LayerReplacementFineTuningExperiment):
    """
    Class to perform the evaluation of a model with unique average layer on some benchmarks, the fine-tuning of the
    model and the evaluation on the same benchmarks again.
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

        target_names = []
        for name, _ in model.named_parameters():
            layer_name = name.split(".")[:-1]
            if layer_name not in target_names:
                target_names.append(layer_name)

        extracted_layers = {}
        get_parameters(model, target_names, extracted_layers, [])

        layers_to_train = {}
        for path, layer in extracted_layers.items():
            if not isinstance(layer, torch.nn.Embedding):
                layers_to_train[tuple(path)] = layer

        return layers_to_train


class LayerReplacementFineTuningAdapterOnTargetsExperiment(LayerReplacementFineTuningExperiment):
    """
    Class to perform the evaluation of a model with unique average layer on some benchmarks, the fine-tuning of the
    model and the evaluation on the same benchmarks again.
    """

    @override
    def prepare_fine_tuning(
            self,
            prepared_models: dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]
    ) -> None:
        for model_key in prepared_models:
            self.log(f"Preparing the model {model_key} for fine-tuning using adapters.")
            model = prepared_models[model_key]
            for parameter in model.parameters():
                parameter.requires_grad = False

            # Wrapping the model with the adapter
            adapted_model = self.get_adapted_model(model)
            prepared_models[model_key] = adapted_model

    def get_adapted_model(
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

        default_dict = {
            "adapter_method": "lora",
            "lora_rank": 16,
            "lora_alpha": 64,
            "target_modules": self.get_label_layers_to_train(model),
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        keys = ["adapter_method", "lora_rank", "lora_alpha", "target_modules", "lora_dropout", "bias", "task_type"]
        config_dict = self.config.get_dict(keys)
        default_dict.update(config_dict)
        config_dict = default_dict

        self.log(f"Creating the adapted model with the following configuration: {config_dict}.")
        try:
            a = get_adapted_model(model, Config.convert_to_config(config_dict))
            print(a)
        except Exception as e:
            print(e)
            raise e

        adapted_model = get_adapted_model(model, Config.convert_to_config(config_dict))
        self.log(f"Adapted model created: {adapted_model}.")

        return adapted_model

    def get_label_layers_to_train(
            self,
            model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel
    ) -> list | str:
        """

        """

        layers_to_train = {}
        get_parameters(model, self.config.get("targets"), layers_to_train, self.config.get("blacklist") if self.config.contains("blacklist") else [])
        target_names = []
        for layer_path in layers_to_train:
            if layer_path[-1] not in target_names:
                target_names.append(layer_path[-1])

        return target_names


class LayerReplacementFineTuningAdapterOnEntireModelExperiment(LayerReplacementFineTuningAdapterOnTargetsExperiment):

    @override
    def get_label_layers_to_train(
            self,
            model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel
    ) -> list | str:
        """
        """

        return "all-linear"