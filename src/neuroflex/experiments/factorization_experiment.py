import copy

import torch

import transformers

from neuroflex.experiments.benchmark_experiment_utils import BenchmarkEvaluation
from neuroflex.utils.factorized_models_utils.factorized_models_utils import get_factorized_model


class FactorizationBenchmarkEvaluation(BenchmarkEvaluation):
    """
    Class to perform the benchmark evaluation on a factorized model.
    """

    mandatory_keys = ["factorization_methods", "target_layers"]

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

        prepared_models = {}

        for factorization_method in self.config.get("factorization_methods"):
            loaded_model = self.load(f"{factorization_method}.pt", "pt")
            prepared_models[f"{factorization_method}"] = loaded_model if loaded_model is not None else get_factorized_model(copy.deepcopy(base_model), self.config).get_model()

        return prepared_models

