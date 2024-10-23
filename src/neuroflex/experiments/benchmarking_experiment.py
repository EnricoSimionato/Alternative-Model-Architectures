import copy
import logging
import os

import matplotlib.pyplot as plt

import torch
import transformers
from typing_extensions import override

from exporch import Config, GeneralPurposeExperiment, get_available_device
from exporch.experiment import evaluate_model_on_benchmark, benchmark_id_metric_name_mapping
from exporch.utils.causal_language_modeling import load_model_for_causal_lm, load_tokenizer_for_causal_lm


class BenchmarkEvaluation(GeneralPurposeExperiment):
    """
    Class to perform the evaluation of a model on some benchmarks.
    """

    mandatory_keys = ["benchmark_ids"]

    @override
    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment.
        It performs the evaluation of the model on some benchmarks.
        """

        already_created_performance_dict = None
        if self.get_data() is not None:
            already_created_performance_dict = self.get_data()[0]
        prepared_models, tokenizer, performance_dict, remaining_analysis = self._prepare_experiment(already_created_performance_dict)
        self._perform_model_evaluation(prepared_models, tokenizer, performance_dict, remaining_analysis)

    def _prepare_experiment(
            self,
            already_created_performance_dict: dict[str, dict[str, dict[str, float]]] = None
    ) -> tuple[
         dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
         transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
         dict[str, dict[str, dict[str, float]]],
         dict[str, list[str]]
    ]:
        """
        Prepares the experiment:
            - Loads the models to be evaluated.
            - Loads the tokenizer.
            - Defines the remaining instances to be processed.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
                The prepared models to be evaluated.
            transformers.AutoTokenizer | transformers.PreTrainedTokenizer:
                The tokenizer of the models.
            dict[str, dict[str, dict[str, float]]]:
                The dictionary containing the performance results.
            dict[str, dict[str, dict[str, float]]]:
                The dictionary containing the remaining analysis to be performed.
        """

        # Initializing the dictionary to store the performance results
        benchmark_ids = self.config.get("benchmark_ids")
        performance_dict = {benchmark_id: {} for benchmark_id in benchmark_ids}

        if already_created_performance_dict is not None:
            performance_dict.update(already_created_performance_dict)
            self.log(f"Previous data loaded.\nLoaded data: {performance_dict}")

        # Loading and preparing the models
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        prepared_models, tokenizer = self._prepare_models()
        model_ids = list(prepared_models.keys())

        # Finding the configurations of the analysis that have not been evaluated yet
        remaining_analysis = {benchmark_id: list(copy.deepcopy(model_ids)) for benchmark_id in benchmark_ids}
        # Evaluating if the analysis has already been done
        for benchmark_id in benchmark_ids:
            for model_key in model_ids:
                if benchmark_id in performance_dict and model_key in performance_dict[benchmark_id]:
                    remaining_analysis[benchmark_id].remove(model_key)
            if len(remaining_analysis[benchmark_id]) == 0:
                del remaining_analysis[benchmark_id]

        return prepared_models, tokenizer, performance_dict, remaining_analysis

    def _perform_model_evaluation(
            self,
            prepared_models: dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            performance_dict: dict[str, dict[str, dict[str, float]]],
            remaining_analysis: dict[str, list[str]],
            performance_dict_storage_slot: int = 0
    ) -> None:
        """
        Performs the evaluation of the model on the benchmarks
        """

        benchmark_ids = self.config.get("benchmark_ids")
        device_str = get_available_device(self.config.get("device") if self.config.contains("device") else "cpu", just_string=True)

        evaluation_args = (self.config.get("evaluation_args") if self.config.contains("evaluation_args")
                           else {benchmark_id: {} for benchmark_id in benchmark_ids})

        for benchmark_id in remaining_analysis.keys():
            # Defining the evaluation parameters
            benchmark_evaluation_args = evaluation_args[benchmark_id]
            self.log(f"Chosen evaluation args for the benchmark {benchmark_id}: {benchmark_evaluation_args}")

            for model_key in remaining_analysis[benchmark_id]:
                model = prepared_models[model_key]
                logging.info(f"Starting the evaluation of the model {model_key} the benchmark {benchmark_id}.")
                print(f"Starting the evaluation of the model {model_key} the benchmark {benchmark_id}.")

                # Evaluating the model
                self.log(f"Starting the evaluation of the model on the device {model.device}.")
                results = evaluate_model_on_benchmark(model, tokenizer, benchmark_id, benchmark_evaluation_args, device_str)
                #results = {benchmark_id: {"acc_norm,none": 0.7}} # Testing
                self.log(f"Results of the model {model_key}: {results}")
                print(f"Results of the model {model_key}: {results}")

                # Storing the results in the dictionary
                performance_dict[benchmark_id][model_key] = results
                self.log(f"Performance dictionary updated with the results.")

                # Moving the model to the CPU to be able to evaluate the next model
                model.cpu()

            # Storing the data
            self.log(f"Trying to store the results on benchmark {benchmark_id}...")
            self.set_data(performance_dict, performance_dict_storage_slot, store=True)
            self.log(f"All results on {benchmark_id} stored.")

        self.log("All data stored.")

        self.log("The evaluation of the models on the benchmarks has been completed.")
        print("The evaluation of the models on the benchmarks has been completed.")

    def _prepare_models(
            self
    ) -> [dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel], transformers.AutoTokenizer | transformers.PreTrainedTokenizer]:
        """
        Gets, stores and returns the prepared models to be evaluated.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
                The prepared models to be evaluated.
        """

        device = self.config.get("device") if self.config.contains("device") else "cpu"
        self.config.set("device", "cpu")

        # Preparing the models
        original_model = self.load("Original Model.pt", "pt")
        if original_model is None:
            original_model = load_model_for_causal_lm(self.config)
        prepared_models = {"Original Model": original_model}
        self.log(f"Original model loaded.")

        tokenizer = self.load("tokenizer.pt", "pt")
        if tokenizer is None:
            tokenizer = load_tokenizer_for_causal_lm(self.config)
        self.log(f"Tokenizer loaded.")

        prepared_models.update(self.prepare_models(copy.deepcopy(original_model), tokenizer))

        self.config.set("device", device)

        for model_key in prepared_models.keys():
            self.log(f"Model {model_key} prepared.")
            self.log(f"Model {model_key} is on device: {prepared_models[model_key].device}")

        # Storing the models
        for model_key, model in prepared_models.items():
            if not self.exists_file(f"{model_key}.pt"):
                self.store(model, f"{model_key}.pt", "pt")
                self.log(f"Model {model_key} stored.")
        # Storing the tokenizer
        if not self.exists_file("tokenizer.pt"):
            self.store(tokenizer, "tokenizer.pt", "pt")
            self.log(f"Tokenizer stored.")

        self.log(f"All models and tokenizer prepared.")

        return prepared_models, tokenizer

    def prepare_models(
            self,
            base_model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer
    ) -> dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
        """
        Returns the prepared models to be evaluated. This method should be implemented by the subclasses if needed.

        Args:
            base_model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The original model to be prepared.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer of the model.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
                The prepared models to be evaluated.
        """

        return {}

    @override
    def _postprocess_results(
            self
    ) -> None:
        """
        Post-processes the results obtained from the experiment.
        It does nothing in this case.
        """

        pass

    @override
    def _plot_results(
            self,
            config: Config,
            data: tuple
    ) -> None:
        """
        Plots the results obtained from the experiment.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
        """

        performance_dict = data[0]

        # Printing the results
        for benchmark_id in performance_dict:
            for model_key in performance_dict[benchmark_id]:
                results = performance_dict[benchmark_id][model_key]
                self.log(f"The performance of the model {model_key} on the benchmark {benchmark_id} is {results}.")
                print(f"The performance of the model {model_key} on the benchmark {benchmark_id} is {results}.")

        figure_size = config.get("figure_size") if config.contains("figure_size") else (10, 15)
        # Plotting histograms of the results
        fig, axes = plt.subplots(1, len(list(performance_dict.keys())), figsize=figure_size)
        if len(list(performance_dict.keys())) == 1:
            axes = [axes]
        for i, benchmark_id in enumerate(performance_dict):
            metric = benchmark_id_metric_name_mapping[benchmark_id]
            axes[i].bar(
                performance_dict[benchmark_id].keys(),
                [model_performance[benchmark_id][metric] for model_performance in performance_dict[benchmark_id].values()],
                width=0.6
            )
            axes[i].set_title(f"Results on {benchmark_id}")


            for rect in axes[i].patches:
                height = rect.get_height()
                axes[i].annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()

        # Storing the plot
        self.store(fig, "results_plot.png", "plt")
