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


class FineTuningExperiment(BenchmarkEvaluation):
    """
    Class to perform the evaluation of a modified model on some benchmarks, the fine-tuning of the model and the
    evaluation on the same benchmark again.
    """

    mandatory_keys = ["task_id", "optimizers_settings", "max_steps"]

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
        self._perform_model_evaluation(prepared_models, tokenizer, performance_dict, remaining_analysis)
        # Fine-tuning the models
        fine_tuned_models, tokenizer = self._perform_fine_tuning(prepared_models, tokenizer)
        # Evaluating the fine-tuned models on the benchmarks
        self._perform_model_evaluation(fine_tuned_models, tokenizer, performance_dict, remaining_analysis)

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

        self.prepare_fine_tuning(prepared_models)

        fine_tuned_models = {model_key: None for model_key in prepared_models}

        # Creating the PyTorch Lightning model
        for model_key in fine_tuned_models:
            base_model = prepared_models[model_key]

            # Creating the dataset
            pl_dataset = get_pytorch_lightning_dataset(
                self.config.get("dataset_id"),
                tokenizer,
                self.config.get("max_len"),
                self.config
            )
            pl_dataset.setup()
            self.config.set("max_steps", len(pl_dataset.train_dataloader()) * self.config.get("max_epochs"))
            print(f"Batch size: {self.config.get('batch_size')}")
            print(f"Max steps: {self.config.get('max_steps')}")
            print(f"Max epochs: {self.config.get('max_epochs')}")
            print(f"Len train dataloader: {len(pl_dataset.train_dataloader())}")
            print(f"Len val dataloader: {len(pl_dataset.val_dataloader())}")
            print(f"Len test dataloader: {len(pl_dataset.test_dataloader())}")

            # Creating the model
            pl_model = get_pytorch_lightning_model(base_model, tokenizer, self.config.get("task_id"), self.config)

            # Creating the trainer
            pl_trainer = get_pytorch_lightning_trainer(self.config.get("task_id"), self.config)

            # Validating the model before training
            #_, validation_results_before_fit = self._validate(pl_model, pl_trainer, pl_dataset)
            # Training the model
            _ = self._fit(pl_model, pl_trainer, pl_dataset)
            # Validating the model after training
            _, fit_validation = self._validate(pl_model, pl_trainer, pl_dataset)
            # Testing the model
            _, fit_test = self._test(pl_model, pl_trainer, pl_dataset)

            fine_tuned_models[model_key] = pl_model.model

        return fine_tuned_models, tokenizer

    def prepare_fine_tuning(
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

        self.create_experiment_directory("checkpoints")
        self.create_experiment_directory("training_logs")

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