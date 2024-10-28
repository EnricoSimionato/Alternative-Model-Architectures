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

from neuroflex.experiments.benchmarking_experiment import BenchmarkEvaluation

from neuroflex.experiments.extratomove import get_parameters


class FineTuningExperiment(BenchmarkEvaluation):
    """
    Class to perform the evaluation of a modified model on some benchmarks, the fine-tuning of the model and the
    evaluation on the same benchmark again.
    """

    mandatory_keys = ["task_id", "optimizers_settings"]

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
        if self.get_data() is not None:
            already_created_performance_dict = self.get_data()[0]
        # Preparing the models, the tokenizer and the performance dictionary
        prepared_models, tokenizer, performance_dict, remaining_analysis = self._prepare_experiment(already_created_performance_dict)

        # Evaluating the models on the benchmarks
        #self._perform_model_evaluation(prepared_models, tokenizer, performance_dict, remaining_analysis, 0)

        # Fine-tuning the models
        fine_tuned_models, tokenizer = self._perform_fine_tuning(prepared_models, tokenizer)
        for model_key in fine_tuned_models:
            self.store(fine_tuned_models[model_key], f"fine_tuned_model_{model_key}", "pt")

        # Evaluating the fine-tuned models on the benchmarks
        self._perform_model_evaluation(fine_tuned_models, tokenizer, performance_dict, remaining_analysis, 1)

        self.log("The experiment has been completed.", print_message=True)

    def _perform_fine_tuning(
         self,
         prepared_models: dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
         tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer
    ) -> tuple[dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel], transformers.AutoTokenizer | transformers.PreTrainedTokenizer]:
        """
        Performs the fine-tuning of the models on a dataset.

        Args:
            prepared_models (dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]):
                The models to fine-tune.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer to use.

        Returns:
            dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
                The fine-tuned models.
            transformers.AutoTokenizer | transformers.PreTrainedTokenizer]:
                The tokenizer used.
        """

        fine_tuned_models = {model_key: None for model_key in prepared_models if model_key != "Original Model"}
        original_model = None
        if self.config.contains("train_original_original_model") and self.config.get("train_original_original_model"):
            fine_tuned_models["Original Model"] = None
        else:
            original_model = prepared_models.pop("Original Model")

        self._prepare_fine_tuning(prepared_models)

        # Creating the dataset
        pl_dataset = get_pytorch_lightning_dataset(
            self.config.get("dataset_id"),
            tokenizer,
            self.config.get("max_len"),
            self.config
        )
        pl_dataset.setup()
        self.config.set("max_steps", len(pl_dataset.train_dataloader()) * self.config.get("max_epochs"))

        if "Original Model" in prepared_models.keys() and "Original Model" not in fine_tuned_models.keys():
            self.log("Evaluating the original model.", print_message=True)
            # Creating the model
            #pl_model = get_pytorch_lightning_model(prepared_models["Original Model"], tokenizer, self.config.get("task_id"), self.config)
            # Creating the trainer
            #pl_trainer = get_pytorch_lightning_trainer(self.config.get("task_id"), self.config)
            # Validating the original model
            #_, validation_results = self._validate(pl_model, pl_trainer, pl_dataset)
            #self.log(validation_results)
            pass
            #prepared_models["Original Model"].cpu()

        # Creating the PyTorch Lightning model
        for model_key in fine_tuned_models:
            base_model = prepared_models[model_key]

            # Creating the model
            pl_model = get_pytorch_lightning_model(base_model, tokenizer, self.config.get("task_id"), self.config)
            self.log(f"Model wrapped with PyTorch Lightning.", print_message=True)

            # Creating the trainer
            pl_trainer = get_pytorch_lightning_trainer(self.config.get("task_id"), self.config)
            self.log(f"PyTorch Lightning Trainer created.", print_message=True)

            # Validating the model before training
            #_, validation_results_before_fit = self._validate(pl_model, pl_trainer, pl_dataset)
            # Training the model
            _ = self._fit(pl_model, pl_trainer, pl_dataset)
            # Validating the model after training
            _, validation_results = self._validate(pl_model, pl_trainer, pl_dataset)
            self.log(validation_results)
            # Testing the model
            _, test_results = self._test(pl_model, pl_trainer, pl_dataset)
            self.log(test_results)

            fine_tuned_models[model_key] = pl_model.model

        if not (self.config.contains("train_original_original_model") and self.config.get("train_original_original_model")):
            fine_tuned_models["Original Model"] = prepared_models["Original Model"]
        else:
            prepared_models["Original Model"] = original_model

        return fine_tuned_models, tokenizer

    def _prepare_fine_tuning(
            self,
            prepared_models: dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]
    ) -> None:
        """
        Prepares the fine-tuning of the models. It does utility operations such as creating the directories to store the
        checkpoints and the training logs.

        Args:
            prepared_models (dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]):
                The models to fine-tune.
        """

        self.log("Preparing the models for fine-tuning.", print_message=True)

        self.create_experiment_directory("checkpoints")
        self.create_experiment_directory("training_logs")

        self.prepare_fine_tuning(prepared_models)

        self.log("Models prepared for fine-tuning.", print_message=True)

        for model_key in prepared_models:
            self.log(f"Model with model key: {model_key}")
            model = prepared_models[model_key]
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.log(f"Parameter {name} is set to trainable.")
                else:
                    self.log(f"Parameter {name} is NOT trainable!")

    def prepare_fine_tuning(
            self,
            prepared_models: dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]
    ) -> None:
        """
        Prepares the fine-tuning of the models. This method can be overridden to add more operations.

        Args:
            prepared_models (dict[str, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]):
                The models to fine-tune.
        """

        self.log("Setting the layers to train, changing the requires_grad attribute to True", print_message=True)
        for model_key in prepared_models:
            model = prepared_models[model_key]
            for parameter in model.parameters():
                parameter.requires_grad = False
            mapping_path_layers_to_train = self.get_layers_to_train(model)
            layers_to_train = mapping_path_layers_to_train.values()
            #self.log(f"Layers to train:\n{'\n'.join(mapping_path_layers_to_train.keys())}")
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

        layers_to_train = {}
        get_parameters(model, self.config.get("targets"), layers_to_train, self.config.get("blacklist") if self.config.contains("blacklist") else [])

        return layers_to_train

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
        self.log("Model fitted.", print_message=True)

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

        pass
