import copy
import gc

import torch

import transformers

from neuroflex.experiments.benchmarking_experiment import BenchmarkEvaluation
from neuroflex.experiments.fine_tuning_experiment import FineTuningExperiment
from neuroflex.utils.factorized_models_utils.factorized_models_utils import get_factorized_model


class FactorizationBenchmarkEvaluation(BenchmarkEvaluation):
    """
    Class to perform the benchmark evaluation on a factorized model.
    """

    mandatory_keys = ["factorization_methods", "target_layers"]

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
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel, None]:
                The prepared models to be evaluated.
        """

        self.log("Preparing factorized models.")

        prepared_models = {}
        try:
            for factorization_method in self.config.get("factorization_methods"):
                loaded_model = self.load(f"{factorization_method}.pt", "pt")
                if loaded_model is not None:
                    prepared_models[f"{factorization_method}"] = loaded_model
                    self.log(f"Factorized model loaded from storage.\nFactorization method: {factorization_method}.", print_message=True)
                else:
                    prepared_models[f"{factorization_method}"] = get_factorized_model(copy.deepcopy(base_model), factorization_method, self.config).get_model()
                    prepared_models[f"{factorization_method}"].approximation_stats = prepared_models[f"{factorization_method}"].compute_approximation_stats()
                    for key in prepared_models[f"{factorization_method}"].approximation_stats:
                        self.log(f"{key}: {prepared_models[f"{factorization_method}"].approximation_stats[key]}", print_message=True)
                    self.log("", print_message=True)

                    self.log(f"Model prepared factorizing the original one.\nFactorization method: {factorization_method}.", print_message=True)
                self.store(prepared_models[f"{factorization_method}"], f"{factorization_method}.pt", "pt")
                if self.is_low_memory_mode():
                    del prepared_models[f"{factorization_method}"]
                    gc.collect()
                    prepared_models[f"{factorization_method}"] = None
                    self.log("Model deleted from memory.", print_message=True)
        except Exception as e:
            self.log(f"Error preparing factorized models: {e}")
            raise e

        return prepared_models


class FactorizationFineTuningExperiment(FineTuningExperiment):
    """
    Class to perform the fine-tuning of a factorized model.
    """

    mandatory_keys = ["factorization_methods", "target_layers"]

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

        self.log(f"Preparing the models using replacement methods {self.config.get('factorization_methods')}.")

        prepared_models = {}
        try:
            for factorization_method in self.config.get("factorization_methods"):
                loaded_model = self.load(f"{factorization_method}.pt", "pt")
                if loaded_model is not None:
                    prepared_models[f"{factorization_method}"] = loaded_model
                    self.log(f"Factorized model loaded from storage..", print_message=True)
                else:
                    self.log(base_model, print_message=True)
                    prepared_models[f"{factorization_method}"] = get_factorized_model(copy.deepcopy(base_model), factorization_method, self.config).get_model()
                    self.log(prepared_models[f"{factorization_method}"], print_message=True)
                    self.log(f"Model prepared factorizing the original one.", print_message=True)
                self.log(f"Factorization method: {factorization_method}.", print_message=True)
                self.store(prepared_models[f"{factorization_method}"], f"{factorization_method}.pt", "pt")
                if self.is_low_memory_mode():
                    del prepared_models[f"{factorization_method}"]
                    gc.collect()
                    prepared_models[f"{factorization_method}"] = None
                    self.log("Model deleted from memory.", print_message=True)
        except Exception as e:
            self.log(f"Error preparing factorized models: {e}")
            raise e

        return prepared_models