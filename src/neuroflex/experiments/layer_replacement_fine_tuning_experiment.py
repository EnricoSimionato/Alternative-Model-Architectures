import copy
from typing import override

import torch

import transformers

from neuroflex.experiments.fine_tuning_experiment import FineTuningExperiment
from neuroflex.pruning.replacement_model import get_layer_replaced_model


class LayerReplacementFineTuningExperiment(FineTuningExperiment):
    """
    Class to perform the evaluation of a model with unique average layer on some benchmarks, the fine-tuning of the
    model and the evaluation on the same benchmarks again.
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

        try:
            for replacement_method in self.config.get("replacement_methods"):
                loaded_model = self.load(f"{replacement_method}.pt", "pt")
                if loaded_model is None:
                    prepared_models[f"{replacement_method}"] = get_layer_replaced_model(
                        copy.deepcopy(base_model), replacement_method, self.get_layers_replacement_mapping(), self.config
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
