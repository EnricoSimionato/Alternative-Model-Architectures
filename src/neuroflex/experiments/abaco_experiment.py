from typing import override

import copy
import gc

import torch
import transformers

from neuroflex.experiments.fine_tuning_experiment import FineTuningExperiment
from neuroflex.abaco.abaco import ABACOModel

class ABACOExperiment(FineTuningExperiment):
    """
    Class to do experiments using ABACO method.
    See neuroflex.abaco.abaco.ABACOModel for more information.
    """

    mandatory_keys = ["adapter_methods", "adapted_layers"]

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

        adapted_layers = self.config.get("adapted_layers").keys()
        fine_tuning_targets = [layer[-1] for layer in self.config.get("fine-tuning_targets")]
        if not all(layer in fine_tuning_targets for layer in adapted_layers) and all(layer in adapted_layers for layer in fine_tuning_targets):
            raise ValueError("Adapted layers and fine-tuning targets must be the same.")

        self.log(f"Preparing the models using replacement methods {self.config.get('adapter_methods')}.")

        prepared_models = {}
        try:
            for adapter_method in self.config.get("adapter_methods"):
                loaded_model = self.load(f"{adapter_method}.pt", "pt")
                if loaded_model is not None:
                    prepared_models[f"{adapter_method}"] = loaded_model
                    self.log(f"ABACO model {adapter_method} loaded from storage..", print_message=True)
                else:
                    if self.config.contains("adapter_method"):
                        raise ValueError("Adapter method already set.")
                    self.config.set("adapter_method", adapter_method)
                    prepared_models[f"{adapter_method}"] = ABACOModel(copy.deepcopy(base_model), self.config, **self.config.get_dict(["initial_alpha", "horizon", "alpha_strategy"]))
                    self.config.set("adapter_method", None)
                    self.log(f"ABACO model {adapter_method} prepared.", print_message=True)
                self.log(f"ABACO method: {adapter_method}.", print_message=True)
                self.store(prepared_models[f"{adapter_method}"], f"{adapter_method}.pt", "pt")
                if self.is_low_memory_mode():
                    del prepared_models[f"{adapter_method}"]
                    gc.collect()
                    prepared_models[f"{adapter_method}"] = None
                    self.log("Model deleted from memory.", print_message=True)
        except Exception as e:
            self.log(f"Error preparing factorized models: {e}")
            raise e

        return prepared_models

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

        self.log("In the case of ABACO trainable elements are already set", print_message=True)

        return prepared_model

    @override
    def _postprocess_fine_tuned_model(
            self,
            fine_tuned_model: ABACOModel
    ) -> ABACOModel:
        """
        Post-processes the fine-tuned model.
        In ABACO the original model is deactivated.

        Args:
            fine_tuned_model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The fine-tuned model.

        Returns:
            torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                The post-processed fine-tuned model.
        """

        self.log("Post-processing the fine-tuned model based on ABACO.", print_message=True)
        fine_tuned_model.deactivate_original_model()

        return fine_tuned_model

