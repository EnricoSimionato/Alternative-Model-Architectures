import torch

import transformers

from neuroflex.experiments.benchmark_experiment_utils import BenchmarkEvaluation
from neuroflex.utils.factorized_models_utils.factorized_models_utils import get_factorized_model


class FactorizationBenchmarkEvaluation(BenchmarkEvaluation):
    """
    Class to perform the benchmark evaluation on a factorized model.
    """

    mandatory_keys = ["factorization_method", "target_layers"]

    def prepare_models(
            self,
            base_model
    ) -> dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
        """
        Returns the prepared models to be evaluated.

        Args:
            base_model:
                The original model to be prepared.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
                The prepared models to be evaluated.
        """

        self.log(f"The prepared model is the model factorized using {self.config.get("factorization_method")}.")
        self.log(f"The prepared models are the original one and the factorized one.")

        loaded_model = self.load(f"{self.config.get('factorization_method')}.pt", "pt")

        return {
            "factorized_model": loaded_model if loaded_model is not None else get_factorized_model(base_model, self.config).get_model(),
            "original_model": base_model,
            #"factorized_model": get_factorized_model(base_model, self.config)
        }
