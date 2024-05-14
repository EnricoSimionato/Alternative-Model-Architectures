from __future__ import annotations

import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from transformers import AutoTokenizer, PreTrainedTokenizer

from gbm.utils.storage_utils import store_model_and_info
from gbm.utils.pipeline.config import Config, ExperimentStatus

from gbm.utils.classification import ClassifierModelWrapper
from gbm.utils.classification.pl_trainer import get_classification_trainer

from gbm.utils.chatbot import ChatbotModelWrapper
from gbm.utils.chatbot.pl_trainer import get_causal_lm_trainer


# TODO documentation of experiment
class Experiment:
    """
    Experiment class to run an experiment.
    It performs the following steps:
    1. Initializes the experiment.
    2. Runs the experiment.
    3. Ends the experiment.

    Args:

    Attributes:

    """

    def __init__(
            self,
            task: str,
            model: nn.Module,
            dataset: LightningDataModule,
            config: Config | str,
            tokenizer: AutoTokenizer | PreTrainedTokenizer = None,
            **kwargs
    ) -> None:
        self.task = task
        self.model = model
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer

        self.lightning_model = None
        self.lightning_trainer = None

    def start_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the experiment.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        self.config.start_experiment()
        print("Experiment started")

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

        if self.config.status is ExperimentStatus.NOT_STARTED:
            self.config.start_experiment()
            print("Experiment started")
        else:
            print("Running the experiment, it is already started")

        self.lightning_trainer = self._get_trainer(**kwargs)
        self.lightning_model = self._get_lightning_model(**kwargs)
        validate_results = self._validate(**kwargs)
        print(f"Validate results before training: {validate_results}")

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
        if self.task == "classification":
            return ClassifierModelWrapper(
                model=self.model,
                tokenizer=self.tokenizer,

                num_classes=self.config.get("num_classes"),
                id2label=self.config.get("id2label"),
                label2id=self.config.get("label2id"),

                learning_rate=self.config.get("learning_rate"),
                max_epochs=self.config.get("num_epochs"),

                warmup_steps=self.config.get("warmup_steps"),
            )

        elif self.task == "question-answering":
            pass
        elif self.task == "chatbot":
            return ChatbotModelWrapper(
                model=self.model,
                tokenizer=self.tokenizer,

                learning_rate=self.config.get("learning_rate"),
                max_epochs=self.config.get("num_epochs"),
                warmup_steps=self.config.get("warmup_steps"),
            )
        else:
            raise ValueError(f"Task {self.task} not recognized")

    def _get_trainer(
            self,
            **kwargs
    ) -> pl.Trainer:
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
        return self.lightning_trainer.fit(
            self.lightning_model,
            self.dataset
        )

    def _validate(
            self,
            **kwargs
    ) -> dict:
        return self.lightning_trainer.validate(
            self.lightning_model,
            self.dataset
        )

    def _test(
            self,
            **kwargs
    ) -> dict:
        return self.lightning_trainer.test(
            self.lightning_model,
            self.dataset
        )

    def _store_experiment(
            self,
            **kwargs
    ) -> None:

        store_model_and_info(
            self.lightning_model.model,
            self.config,
            tokenizer=self.tokenizer
        )


if __name__ == "__main__":
    """
    configuration = Config("/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_LOCAL.json")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    from gbm.utils.classification import IMDBDataModule
    from transformers import AutoModelForSequenceClassification
    experiment = Experiment(
        task="classification",
        model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased"),
        dataset=IMDBDataModule(
            configuration.get("batch_size"),
            configuration.get("num_workers"),
            tokenizer,
            configuration.get("max_len_tokenizer"),
            configuration.get("split")
        ),
        config=configuration,
        tokenizer=tokenizer
    )

    experiment.run_experiment()
    """

    configuration = Config("/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_LOCAL.json")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer_mistral.pad_token = tokenizer_mistral.eos_token
    tokenizer_mistral.padding_side = "right"
    tokenizer.pad_token = tokenizer_mistral.pad_token
    tokenizer.padding_side = tokenizer_mistral.padding_side

    from gbm.utils.chatbot import OpenAssistantGuanacoDataModule
    from transformers import AutoModelForCausalLM

    experiment = Experiment(
        task="chatbot",
        model=AutoModelForCausalLM.from_pretrained("bert-base-uncased"),
        dataset=OpenAssistantGuanacoDataModule(
            configuration.get("batch_size"),
            configuration.get("num_workers"),
            tokenizer_mistral,
            configuration.get("max_len_tokenizer"),
            configuration.get("split")
        ),
        config=configuration,
        tokenizer=tokenizer
    )

    experiment.run_experiment()
