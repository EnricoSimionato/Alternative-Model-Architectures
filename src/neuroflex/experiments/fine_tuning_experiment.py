import gc
from typing import Any, override

import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl

import transformers

from exporch import Config, get_available_device
from exporch.experiment import benchmark_id_metric_name_mapping
from exporch.utils.general_framework_utils.model_dispatcher import get_pytorch_lightning_model
from exporch.utils.general_framework_utils.trainer_dispatcher import get_pytorch_lightning_trainer
from exporch.utils.general_framework_utils.datamodule_dispatcher import get_pytorch_lightning_dataset
from exporch.utils.model_utils import get_parameters

from neuroflex.experiments.benchmarking_experiment import BenchmarkEvaluation


class FineTuningExperiment(BenchmarkEvaluation):
    """
    Class to perform the evaluation of a modified model on some benchmarks, the fine-tuning of the model and the
    evaluation on the same benchmark again.
    """

    mandatory_keys = ["task_id", "optimizers_settings", "fine-tuning_targets"]
    fine_tuned_models_prefix = "fine_tuned_model_"

    @override
    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment.
        The experiment consists of the following steps:
            - Prepare the models and the tokenizer.
            - Evaluate the models on the benchmarks.
            - Fine-tune the models.
            - Evaluate the fine-tuned models on the benchmarks.
        """

        # Checking if the experiment has already been run and retrieving the data
        already_created_performance_dict = None
        data = self.get_data()
        if data is not None and len(data) > 0:
            already_created_performance_dict = self.get_data()[0]
        # Preparing the models, the tokenizer and the performance dictionary
        prepared_models, tokenizer, performance_dict, remaining_analysis = self._prepare_experiment(already_created_performance_dict, None)

        # Evaluating the models on the benchmarks
        self._perform_model_evaluation(prepared_models, tokenizer, performance_dict, remaining_analysis, 0)

        # Fine-tuning the models
        fine_tuned_models, tokenizer = self._perform_fine_tuning(prepared_models, tokenizer)

        already_created_performance_dict = None
        data = self.get_data()
        if data is not None and len(data) > 1:
            already_created_performance_dict = self.get_data()[1]
        fine_tuned_models, tokenizer, performance_dict, remaining_analysis = self._prepare_experiment(already_created_performance_dict, fine_tuned_models)
        # Evaluating the fine-tuned models on the benchmarks
        self._perform_model_evaluation(fine_tuned_models, tokenizer, performance_dict, remaining_analysis, 1)

        # Cleaning the storage if needed
        self.clean_storage()
        self.log("The experiment has been completed.", print_message=True)

    def _perform_fine_tuning(
         self,
         prepared_models: dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None],
         tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer
    ) -> tuple[dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None], transformers.AutoTokenizer | transformers.PreTrainedTokenizer]:
        """
        Performs the fine-tuning of the models on a dataset.

        Args:
            prepared_models (dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None]):
                The models to fine-tune.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer to use.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None]:
                The fine-tuned models.
            transformers.AutoTokenizer | transformers.PreTrainedTokenizer]:
                The tokenizer used.
        """

        original_model = prepared_models.pop("Original Model")
        fine_tuned_models = {f"{self.fine_tuned_models_prefix}{model_key}": None for model_key in prepared_models}

        self._prepare_utils_for_fine_tuning()

        # Creating the dataset
        dataset_id = self.config.get("dataset_id")
        pl_dataset = get_pytorch_lightning_dataset(dataset_id, tokenizer, self.config.get("max_len"), self.config)
        pl_dataset.setup()
        #self.config.set("max_steps", len(pl_dataset.train_dataloader()) * self.config.get("max_epochs"))

        # Fine-tuning the models
        all_models_already_fine_tuned = True
        for model_key in list(prepared_models.keys()):
            self.log(f"Fine-tuning model with model key: {model_key}.", print_message=True)
            model = prepared_models[model_key]

            # Loading the model if we are in low memory mode
            if self.is_low_memory_mode() and model is None:
                model = self.load(f"{model_key}.pt", "pt")
                if model is None:
                    raise ValueError(f"Model {model_key} not found in storage.")
            already_fine_tuned = self._prepare_fine_tuning(model_key, model)
            all_models_already_fine_tuned = all_models_already_fine_tuned and already_fine_tuned
            model.to(get_available_device(self.config.get("device") if self.config.contains("device") else "cpu"))

            if not already_fine_tuned:
                self.log(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", print_message=True)
                # Creating the Lightning model
                pl_model = self.get_pytorch_lightning_model(model, tokenizer, self.config.get("task_id"), self.config)
                self.log(f"Model wrapped with PyTorch Lightning.", print_message=True)

                # Creating the Lightning trainer
                pl_trainer = self.get_pytorch_lightning_trainer(self.config.get("task_id"), self.config)
                self.log(f"PyTorch Lightning Trainer created.", print_message=True)

                # Validating the model before training
                _, validation_results_before_fit = self._validate(pl_model, pl_trainer, pl_dataset)
                self.log(f"Validation results before fit:\n {validation_results_before_fit}")

                # Training the model
                try:
                    _ = self._fit(pl_model, pl_trainer, pl_dataset)
                except (KeyboardInterrupt, SystemExit, RuntimeError) as e:
                    self.log(f"Training interrupted by the user. Exception: {e}", print_message=True)

                fine_tuned_model = pl_model.model
                # Storing the fine-tuned model
                self.store(fine_tuned_model, f"{self.fine_tuned_models_prefix}{model_key}.pt", "pt")

                # Post-processing after training before validation and testing
                fine_tuned_model = self._postprocess_fine_tuned_model(fine_tuned_model)

                # Validating the model after training
                _, validation_results = self._validate(pl_model, pl_trainer, pl_dataset)
                self.log(f"Validation results after fit:\n {validation_results}")
                # Testing the model
                # For now testing is disabled
                #_, test_results = self._test(pl_model, pl_trainer, pl_dataset)
                #self.log(f"Test results:\n {test_results}")

            else:
                fine_tuned_model = model

            fine_tuned_models[f"{self.fine_tuned_models_prefix}{model_key}"] = fine_tuned_model

            self.log(f"Model with model key: {model_key} fine-tuned.", print_message=True)
            self.log(f"{fine_tuned_models[f"{self.fine_tuned_models_prefix}{model_key}"]}")

            # Deleting the model from memory if we are in low memory mode
            if self.is_low_memory_mode():
                del fine_tuned_models[f"{self.fine_tuned_models_prefix}{model_key}"]
                del prepared_models[model_key]
                gc.collect()
                fine_tuned_models[f"{self.fine_tuned_models_prefix}{model_key}"] = None

        # Evaluating the original model if there is at least one model that has not been fine-tuned
        if not all_models_already_fine_tuned:
            # Loading the original model if it is not in memory
            if original_model is None:
                original_model = self.load("Original Model.pt", "pt")
                if original_model is None:
                    raise ValueError("Original Model not found in storage.")
            self.log("Evaluating the original model.", print_message=True)
            # Creating the model
            pl_model = self.get_pytorch_lightning_model(original_model, tokenizer, self.config.get("task_id"), self.config)
            # Creating the trainer
            pl_trainer = self.get_pytorch_lightning_trainer(self.config.get("task_id"), self.config)
            # Validating the original model
            _, validation_results = self._validate(pl_model, pl_trainer, pl_dataset)
            self.log(validation_results)
            del original_model
            gc.collect()

        return fine_tuned_models, tokenizer

    def _prepare_utils_for_fine_tuning(
            self
    ) -> None:
        """
        Prepares the utilities for the fine-tuning of the models.
        """

        self.create_experiment_directory("checkpoints")
        self.create_experiment_directory("training_logs")

    def _prepare_fine_tuning(
            self,
            model_key: str,
            prepared_model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None
    ) -> bool:
        """
        Prepares the fine-tuning of the given model.
        It checks if the model has already been fine-tuned:
            - If it has been fine-tuned, it loads the fine-tuned model (unless we are in low memory mode).
            - If it has not been fine-tuned, it prepares the model for fine-tuning by setting the layers to train.

        Args:
            model_key (str):
                The key of the model.
            prepared_model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel | None):
                The model to fine-tune.

        Returns:
            bool:
                True if the model has already been fine-tuned, False otherwise.
        """

        self.log("Checking if the model have already been fine-tuned.", print_message=True)
        already_fine_tuned = model_key
        fine_tuned_model = self.load("fine_tuned_model_" + model_key + ".pt", "pt")
        if fine_tuned_model is not None:
            if self.is_low_memory_mode():
                del fine_tuned_model
            self.log("Model with model key: " + model_key + " has already been fine-tuned.", print_message=True)
        else:
            already_fine_tuned = False
            self.log("Model with model key: " + model_key + " has not been fine-tuned.", print_message=True)

        if not already_fine_tuned:
            self.log("Preparing the model for fine-tuning.", print_message=True)

            # Preparing the model for fine-tuning using the method depending on the type of experiment we are performing
            prepared_model = self.prepare_fine_tuning(prepared_model)

            self.log("Model prepared for fine-tuning.", print_message=True)

            for name, param in prepared_model.named_parameters():
                if param.requires_grad:
                    self.log(f"Parameter {name} is set to trainable.")
                else:
                    self.log(f"Parameter {name} is NOT trainable!")

        return already_fine_tuned

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

        self.log("Setting the layers to train, changing the requires_grad attribute to True", print_message=True)

        for parameter in prepared_model.parameters():
            parameter.requires_grad = False

        mapping_path_layers_to_train = self.get_layers_to_train(prepared_model)
        layers_to_train = mapping_path_layers_to_train.values()
        for layer in layers_to_train:
            try:
                layer.weight.requires_grad = True
            except AttributeError as e:
                self.log(f"Error setting the layer {layer} to trainable, it does not have the attribute weight.")
                raise e
            try:
                layer.bias.requires_grad = True
            except AttributeError:
                    self.log(f"Error setting the layer {layer} to trainable, it does not have the attribute bias.")
                    self.log("Continuing the process.")

        return prepared_model

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

        if model is None:
            raise ValueError("There are no layers to train. The model cannot be None.")

        # Getting the layers to train
        layers_to_train = {}
        get_parameters(model, self.config.get("fine-tuning_targets"), layers_to_train, self.config.get("blacklist") if self.config.contains("blacklist") else [])

        return layers_to_train

    def get_pytorch_lightning_model(
            self,
            model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            task_id: str,
            config: Config
    ) -> pl.LightningModule:
        """
        Returns the PyTorch Lightning model to use.

        Returns:
            pl.LightningModule:
                The PyTorch Lightning model to use.
        """

        self.log("Getting the PyTorch Lightning model.", print_message=True)
        return get_pytorch_lightning_model(model, tokenizer, task_id, config)

    def get_pytorch_lightning_trainer(
            self,
            task_id: str,
            config: Config,
    ) -> pl.Trainer:
        """
        Returns the PyTorch Lightning trainer to use.

        Args:
            task_id (str):
                The task ID.
            config (Config):
                The configuration of the experiment.

        Returns:
            pl.Trainer:
                The PyTorch Lightning trainer to use.
        """

        self.log("Getting the PyTorch Lightning trainer.", print_message=True)
        return get_pytorch_lightning_trainer(task_id, config)

    def _fit(
            self,
            pl_model,
            pl_trainer,
            pl_dataset
    ) -> pl.LightningModule:
        """
        Trains the model on the dataset.

        Args:
            pl_model (pl.LightningModule):
                The model to train.
            pl_trainer (pl.Trainer):
                The trainer to use.
            pl_dataset (pl.LightningDataModule):
                The dataset to use.

        Returns:
            pl.LightningModule:
                The trained model.
        """

        self.log("Fitting the model.", print_message=True)
        pl_trainer.fit(
            pl_model,
            pl_dataset
        )
        self.log("Training ended. Model fitted.", print_message=True)

        return pl_model

    def _validate(
            self,
            pl_model,
            pl_trainer,
            pl_dataset
    ) -> tuple[pl.LightningModule, Any]:
        """
        Trains the model on the dataset.

        Args:
            pl_model (pl.LightningModule):
                The model to evaluate.
            pl_trainer (pl.Trainer):
                The trainer to use.
            pl_dataset (pl.LightningDataModule):
                The dataset to use.

        Returns:
            pl.LightningModule:
                The evaluated model.
            Any:
                The results obtained from the validation.
        """

        self.log("Validating the model.", print_message=True)
        results = pl_trainer.validate(
            pl_model,
            pl_dataset
        )
        self.log("Model validated.", print_message=True)

        return pl_model, results

    def _test(
            self,
            pl_model,
            pl_trainer,
            pl_dataset
    ) -> tuple[pl.LightningModule, Any]:
        """
        Trains the model on the dataset.

        Args:
            pl_model (pl.LightningModule):
                The model to test.
            pl_trainer (pl.Trainer):
                The trainer to use.
            pl_dataset (pl.LightningDataModule):
                The dataset to use.

        Returns:
            pl.LightningModule:
                The tested model.
            Any:
                The results obtained from the test.
        """

        self.log("Testing the model", print_message=True)
        results = pl_trainer.test(
            pl_model,
            pl_dataset
        )
        self.log("Model tested.", print_message=True)

        return pl_model, results

    def _postprocess_fine_tuned_model(
            self,
            fine_tuned_model: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel
    ) -> torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
        """
        Post-processes the fine-tuned model.

        Args:
            fine_tuned_model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The fine-tuned model.

        Returns:
            torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                The post-processed fine-tuned model.
        """

        return fine_tuned_model

    def clean_storage(
            self
    ) -> None:
        """
        Cleans the storage of the experiment.
        """

        self.delete("Original Model.pt")
        self.delete("tokenizer.pt")

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
            data (tuple):
                The data obtained from the analysis, containing initial and fine-tuned performances, training losses, and validation losses.
        """

        def merge_benchmark_performance(
                pre_finetuning_performance: dict,
                post_finetuning_performance: dict
        ) -> dict:
            merged_dict = {}

            # Get all unique benchmarks
            benchmarks = set(pre_finetuning_performance.keys()).union(post_finetuning_performance.keys())

            for benchmark in benchmarks:
                merged_dict[benchmark] = {}

                pre_finetuning_performance_benchmark = pre_finetuning_performance.get(benchmark, {})
                post_finetuning_performance_benchmark = post_finetuning_performance.get(benchmark, {})

                # Get all unique models for the current benchmark
                model_ids = set(
                    pre_finetuning_performance_benchmark.keys()
                ).union(
                    model_id.replace(self.fine_tuned_models_prefix, "") for model_id in post_finetuning_performance_benchmark.keys()
                )

                for model_id in model_ids:
                    # Fetch performance from each dictionary or None if not present
                    initial_performance = pre_finetuning_performance_benchmark.get(model_id)
                    final_performance = post_finetuning_performance_benchmark.get(self.fine_tuned_models_prefix + model_id) if self.fine_tuned_models_prefix + model_id in post_finetuning_performance_benchmark else post_finetuning_performance_benchmark.get(model_id)

                    # Structure as required
                    merged_dict[benchmark][model_id] = {
                        "initial": initial_performance,
                        "final": final_performance
                    }

            return merged_dict

        performance_dict = merge_benchmark_performance(data[0], data[1])
        # Printing the results
        self.log("\n------------------------------------------------------------------------------", print_message=True)
        self.log("The performance of the models on the benchmarks before training is as follows:\n", print_message=True)
        for benchmark_id in list(performance_dict.keys()):
            for model_key in list(performance_dict[benchmark_id].keys()):
                results = performance_dict[benchmark_id][model_key]
                keys = list(results.keys())
                for key in keys:
                    metric_value = results[key][benchmark_id][benchmark_id_metric_name_mapping[benchmark_id]] if results[key] else None
                    self.log(f"{key[0].upper()+key[1:]} performance of model {model_key} on the benchmark "
                             f"{benchmark_id} -> \t\t{benchmark_id_metric_name_mapping[benchmark_id]}: {metric_value}.",
                             print_message=True)
                self.log("", print_message=True)
            self.log("                    --------------------------------------                    ", print_message=True)
        self.log("------------------------------------------------------------------------------\n", print_message=True)

        # Plotting histograms of the results
        figure_size = config.get("figure_size") if config.contains("figure_size") else (10, 15)
        fig, axes = plt.subplots(1, len(list(performance_dict.keys())), figsize=figure_size)
        if len(list(performance_dict.keys())) == 1:
            axes = [axes]
        for i, benchmark_id in enumerate(list(performance_dict.keys())):
            metric = benchmark_id_metric_name_mapping[benchmark_id]
            initial_performance, final_performance, model_labels = ([], [], [])

            for model_key, performances in performance_dict[benchmark_id].items():
                if performances["initial"]:
                    initial_performance.append(performances["initial"][benchmark_id].get(metric, 0))
                else:
                    initial_performance.append(0)

                if performances["final"]:
                    final_performance.append(performances["final"][benchmark_id].get(metric, 0))
                else:
                    final_performance.append(0)

                model_labels.append(model_key)

            # Plotting both initial and final results
            bar_width = 0.35
            x = range(len(model_labels))

            axes[i].bar([p - bar_width / 2 for p in x], initial_performance, width=bar_width, label="Before Tine-Tuning")
            axes[i].bar([p + bar_width / 2 for p in x], final_performance, width=bar_width, label="After Fine-Tuning")

            # Adding labels and titles
            axes[i].set_title(f"Results on {benchmark_id}")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(model_labels)
            axes[i].legend()

            for rect in axes[i].patches:
                height = rect.get_height()
                axes[i].annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()

        # Storing the plot of the benchmark results
        self.store(fig, "results_on_benchmark.pdf", "plt")
        self.log("Benchmark performance plot saved.", print_message=True)

        # TODO Add the training and validation loss plots
        """
        training_loss_dict = {
            "model_1": [0.9, 0.7, 0.6, 0.5],
            "model_2": [1.0, 0.8, 0.7, 0.6],
            # Additional models...
        }

        validation_loss_dict = {
            "model_1": [0.85, 0.75, 0.65, 0.55],
            "model_2": [0.95, 0.85, 0.75, 0.65],
            # Additional models...
        }

        original_validation_loss = 0.78
        #training_loss_dict, validation_loss_dict, config, original_validation_loss = data[2:]

        losses_figure_size = config.get("losses_figure_size") if config.contains("losses_figure_size") else (12, 8)
        fig, ax = plt.subplots(1, 1, figsize=losses_figure_size)

        for model_key in training_loss_dict:
            ax.plot(training_loss_dict[model_key], label=f"{model_key} - Training Loss", linestyle="--")
            ax.plot(validation_loss_dict[model_key], label=f"{model_key} - Validation Loss", linestyle="-")

        max_steps = max(
            max(len(losses) for losses in training_loss_dict.values()),
            max(len(losses) for losses in validation_loss_dict.values())
        )
        # Plotting original model validation loss if available
        if original_validation_loss:
            ax.plot([original_validation_loss] * len(validation_loss_dict[model_key]),
                    label="Original Model Validation Loss", linestyle=":")

        ax.set_title("Training and Validation Loss during Fine-Tuning")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        plt.tight_layout()
        fig_path = "training_validation_loss.pdf"
        plt.savefig(fig_path)
        print(f"Training and validation loss plot saved to {fig_path}")
        plt.show()
        """

# Not implemented yet
class FineTuningExperimentUsingAdapter(FineTuningExperiment):
    pass