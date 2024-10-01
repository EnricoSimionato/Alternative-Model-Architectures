import copy
import logging
import os

import torch
import transformers

from exporch import GeneralPurposeExperiment, get_available_device
from exporch.experiment import evaluate_model_on_benchmark
from exporch.utils.causal_language_modeling import load_model_for_causal_lm, load_tokenizer_for_causal_lm


class BenchmarkEvaluation(GeneralPurposeExperiment):
    """
    Class to perform the evaluation of a model on some benchmarks.
    """

    mandatory_keys = ["benchmark_ids"]

    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment.
        It performs the evaluation of the model on some benchmarks.
        """

        self._perform_model_evaluation()
        self._plot_results()

    def _perform_model_evaluation(
            self
    ) -> None:
        """
        Performs the evaluation of the model on the benchmarks
        """

        config = self.config

        # Initializing the dictionary to store the performance results
        benchmark_ids = config.get("benchmark_ids")
        performance_dict = {benchmark_id: {} for benchmark_id in benchmark_ids}
        remaining_benchmark_ids = benchmark_ids

        if self.data is not None:
            already_created_performance_dict = self.data
            performance_dict.update(already_created_performance_dict)
            self.log(f"Previous data loaded.\nLoaded data: {performance_dict}")
            analyzed_benchmark_ids = list(benchmark_id for benchmark_id in performance_dict.keys() if len(performance_dict[benchmark_id]) > 0)
            remaining_benchmark_ids = list(set(benchmark_ids) - set(analyzed_benchmark_ids))

            if len(remaining_benchmark_ids) == 0:
                self.log(f"Computation is not needed, the analysis has already been performed.")
                return

        # Getting the parameters from the configuration
        device_str = get_available_device(config.get("device") if config.contains("device") else "cpu", just_string=True)
        self.config.set("device", "cpu")
        evaluation_args = (config.get("evaluation_args") if config.contains("evaluation_args")
                           else {benchmark_id: {} for benchmark_id in benchmark_ids})

        # Loading the model and the tokenizer
        base_model = load_model_for_causal_lm(config)
        self.log(f"Model loaded.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = load_tokenizer_for_causal_lm(config)
        self.log(f"Tokenizer loaded.")

        # Preparing the model
        prepared_models = self.prepare_models(copy.deepcopy(base_model))
        self.log(f"Models prepared.")

        # Storing the models
        for model_key, model in prepared_models.items():
            self.store(model, f"{model_key}.pt", "pt")
        # Storing the tokenizer
        self.store(tokenizer, "tokenizer.pt", "pt")

        self.config.set("device", device_str)
        for benchmark_id in remaining_benchmark_ids:
            for model_key, model in prepared_models.items():
                logging.info(f"Starting the evaluation of the model {model_key} the benchmark: {benchmark_id}.")
                print(f"Starting the evaluation of the model {model_key} the benchmark: {benchmark_id}.")

                # Defining the evaluation parameters
                benchmark_evaluation_args = evaluation_args[benchmark_id]
                self.log(f"Chosen evaluation args: {benchmark_evaluation_args}")

                # Evaluating the model
                self.log(f"Starting the evaluation of the model on the device {model.device}.")
                results = evaluate_model_on_benchmark(model, tokenizer, benchmark_id, benchmark_evaluation_args, device_str)
                # results = {benchmark_id: {"acc_norm,none": 0.7}} # Testing
                self.log(f"Results of the model {model_key}: {results}")
                print(f"Results of the model {model_key}: {results}")

                # Storing the results in the dictionary
                performance_dict[benchmark_id][model_key] = results
                self.log(f"Performance dictionary updated with the results.")

            # Storing the data
            self.data = (performance_dict,)
            self.log(f"Trying to store the results on benchmark {benchmark_id}...")
            self.store_data()
            self.log(f"All results on {benchmark_id} stored.")

        self.log("All data stored.")

        self.log("The analysis has been completed.")
        print("The analysis has been completed.")

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

        self.log(f"The prepared model is the original one.")

        return {"original_model": base_model}

    def _plot_results(
            self,

    ) -> None:
        """
        Plots the results of the experiment.
        """

        performance_dict = self.data[0]

        for benchmark_id in performance_dict:
            for model_key in performance_dict[benchmark_id]:
                results = performance_dict[benchmark_id][model_key]
                self.log(f"The performance of the model {model_key} on the benchmark {benchmark_id} is {results}.")
                print(f"The performance of the model {model_key} on the benchmark {benchmark_id} is {results}.")