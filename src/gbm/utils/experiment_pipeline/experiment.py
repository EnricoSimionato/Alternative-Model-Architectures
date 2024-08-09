from __future__ import annotations

import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

import transformers
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer
)

from gbm.utils.experiment_pipeline.storage_utils import (
    store_model_and_info,
    load_model_and_info
)

from gbm.utils.experiment_pipeline.config import (
    ExperimentStatus,
    Config
)

from gbm.utils.classification import ClassifierModelWrapper
from gbm.utils.classification.pl_trainer import get_classification_trainer

from gbm.utils.chatbot import ChatbotModelWrapper
from gbm.utils.chatbot.pl_trainer import get_causal_lm_trainer


def get_path_to_experiments(
        environment: str,
        **kwargs
) -> str:
    """
    Returns the path to the experiments.

    Args:
        environment (str):
            The environment where to run the experiments.
        kwargs:
            Additional keyword arguments.

    Returns:
        str:
            The path to the experiments.
    """

    if environment == "local":
        base_path = ("/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/"
                     "performed_experiments/")
    elif environment == "server":
        base_path = "/home/enricosimionato/thesis/Alternative-Model-Architectures/src/experiments/performed_experiments"
    elif environment == "colab":
        base_path = "/content/Alternative-Model-Architectures/src/experiments/performed_experiments"
    else:
        raise ValueError("Invalid environment. Choose either 'server' or 'local'.")

    return base_path


class Experiment:
    """
    Class to run an experiment in a standardized way.

    The experiment comprises the following steps:
    1. Experiment initialization:
        - Starts the experiment defining the paths where to store the configuration, the model, the logs, and the
          checkpoints.
    2. Experiment execution:
        - Initializes the PyTorch Lightning model and trainer.
        - Validates the model before training.
        - Trains the model.
        - Validates the model after training.
        - Tests the model.
    3. Experiment storage:
        - Stores the model and the configuration.

    Args:
        task (str):
            The task to perform.
        model (nn.Module):
            The model to use.
        dataset (LightningDataModule):
            The dataset to use.
        config (Config):
            The configuration containing the information about the experiment.
        tokenizer (AutoTokenizer | PreTrainedTokenizer, optional):
            The tokenizer to use.
        store_model_function (callable, optional):
            The function to store the model.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        task (str):
            The task to perform.
        model (nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
            The model to use.
        dataset (LightningDataModule):
            The dataset to use.
        config (Config):
            The configuration containing the information about the experiment.
        tokenizer (AutoTokenizer | PreTrainedTokenizer):
            The tokenizer to use.
        lightning_model (pl.LightningModule):
            The PyTorch Lightning model.
        lightning_trainer (pl.Trainer):
            The PyTorch Lightning trainer.
        store_model_function (callable):
            The function to store the model.
    """

    def __init__(
            self,
            task: str,
            model: [nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            dataset: LightningDataModule,
            config: Config,
            tokenizer: AutoTokenizer | PreTrainedTokenizer = None,
            store_model_function: callable = None,
            **kwargs
    ) -> None:
        self.task = task
        self.model = model
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer

        self.lightning_model = None
        self.lightning_trainer = None

        self.store_model_function = store_model_function

        config.set("task", task)

    def start_experiment(
            self,
            **kwargs
    ) -> dict:
        """
        Initializes the experiment.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the paths where to store the configuration, the model, the logs, and the
                checkpoints.
        """

        if self.config.status is ExperimentStatus.NOT_STARTED:
            self.config.start_experiment()
            print("Experiment started")
        else:
            print("Running the experiment, it is already started")

        paths_dict = self.config.get_paths()

        return paths_dict

    def run_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Runs the experiment.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        self.start_experiment(**kwargs)

        self.dataset.setup()

        self.lightning_trainer = self._get_trainer(**kwargs)
        self.lightning_model = self._get_lightning_model(**kwargs)
        #validate_results = self._validate(**kwargs)
        #print(f"Validate results before training: {validate_results}")

        fit_results = self._fit(**kwargs)
        print(f"Fit results: {fit_results}")
        validate_results = self._validate(**kwargs)
        print(f"Validate results: {validate_results}")
        test_results = self._test(**kwargs)
        print(f"Test results: {test_results}")

        self.config.end_experiment()

        self._store_experiment()

        print("Experiment completed")

    def _get_lightning_model(
            self,
            **kwargs
    ) -> pl.LightningModule:
        """
        Returns the PyTorch Lightning model to train the model using the PyTorch Lightning framework.
        The model is defined based on the task to perform.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            pl.LightningModule:
                The PyTorch Lightning model.
        """
        pl_model = None

        # Defining training arguments
        training_keys = ["optimizers_settings"]
        training_args = {}
        for key in training_keys:
            if self.config.contains(key):
                training_args[key] = self.config.get(key)

        if self.config.contains("max_epochs"):
            training_args["max_steps"] = self.config.get("max_epochs") * len(self.dataset.train_dataloader())
        else:
            training_args["max_steps"] = len(self.dataset.train_dataloader())

        # Defining the data type of the model
        dtype = (self.config.get("dtype") if self.config.contains("dtype") else "float32")

        if self.task == "classification":
            pl_model = ClassifierModelWrapper(
                model=self.model,
                tokenizer=self.tokenizer,

                num_classes=self.config.get("num_classes"),
                id2label=self.config.get("id2label"),
                label2id=self.config.get("label2id"),

                **training_args,

                path_to_storage=self.config.get("path_to_experiment"),
                dtype=dtype
            )

        elif self.task == "question-answering":
            pass
        elif self.task == "chatbot":
            pl_model = ChatbotModelWrapper(
                model=self.model,
                tokenizer=self.tokenizer,

                **training_args,

                stop_tokens=self.config.get("stop_tokens"),

                path_to_storage=self.config.get("path_to_experiment"),
                dtype=dtype
            )

        else:
            raise ValueError(f"Task {self.task} not recognized")

        return pl_model

    def _get_trainer(
            self,
            **kwargs
    ) -> pl.Trainer:
        """
        Returns the PyTorch Lightning trainer to train the model using the PyTorch Lightning framework.
        The trainer is defined based on the task to perform.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            pl.Trainer:
                The PyTorch Lightning trainer.
        """

        if self.task == "classification":
            return get_classification_trainer(
                self.config,
                **kwargs
            )
        elif self.task == "question-answering":
            pass
        elif self.task == "chatbot":
            return get_causal_lm_trainer(
                self.config,
                **kwargs
            )
        else:
            raise ValueError(f"Task {self.task} not recognized")

    def _fit(
            self,
            **kwargs
    ) -> dict:
        """
        Trains the model using the PyTorch Lightning framework.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the results of the training.
        """

        return self.lightning_trainer.fit(
            self.lightning_model,
            self.dataset
        )

    def _validate(
            self,
            **kwargs
    ) -> dict:
        """
        Validates the model using the PyTorch Lightning framework.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the results of the validation.
        """

        return self.lightning_trainer.validate(
            self.lightning_model,
            self.dataset
        )

    def _test(
            self,
            **kwargs
    ) -> dict:
        """
        Tests the model using the PyTorch Lightning framework.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the results of the testing.
        """

        return self.lightning_trainer.test(
            self.lightning_model,
            self.dataset
        )

    def _store_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Stores the model and the configuration.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        store_model_and_info(
            self.model,
            self.config,
            tokenizer=self.tokenizer,
            store_model_function=self.store_model_function
        )

    def store_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Stores the model and the configuration.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        self._store_experiment()

    @classmethod
    def load_experiment(
            cls,
            path_to_experiment: str,
            load_model_function: callable = None,
            **kwargs
    ) -> Experiment:
        """
        Loads the model and the configuration.

        Args:
            path_to_experiment (str):
                The path to the experiment.
            load_model_function (callable, optional):
                The function to load the model. Defaults to None.
            kwargs:
                Additional keyword arguments.

        Returns:
            Experiment:
                The loaded experiment.
        """

        model, config, tokenizer = load_model_and_info(
            path_to_experiment,
            load_model_function=load_model_function
        )

        experiment = cls(
            task=config.get("task"),
            model=model,
            dataset=None,
            config=config,
            tokenizer=tokenizer
        )

        return experiment

    def get_model(
            self
    ) -> nn.Module:
        """
        Returns the model.

        Returns:
            nn.Module:
                The model.
        """

        return self.model

    def get_config(
            self
    ) -> Config:
        """
        Returns the configuration.

        Returns:
            Config:
                The configuration.
        """

        return self.config

    def get_tokenizer(
            self
    ) -> AutoTokenizer | PreTrainedTokenizer:
        """
        Returns the tokenizer.

        Returns:
            AutoTokenizer | PreTrainedTokenizer:
                The tokenizer.
        """

        return self.tokenizer
