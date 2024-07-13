from __future__ import annotations

from typing import Any

import torch
import torchmetrics

import pytorch_lightning as pl

import transformers

from gbm.models.global_dependent_model import RegularizedTrainingInterface
from gbm.utils.classification.pl_metrics import ClassificationStats
from gbm.utils.pl_utils.utility_mappings import optimizers_mapping


class ClassifierModelWrapper(pl.LightningModule):
    """
    Wrapper to train a classifier model in Pytorch Lightning.

    Args:
        model (nn.Module):
            The model to wrap.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer used to tokenize the inputs.
        num_classes (int):
            Number of classes of the problem.
        id2label (dict):
            Mapping from class IDs to labels.
        label2id (dict):
            Mapping from labels to class IDs.
        optimizers_settings (list[dict]):
            List of dictionaries containing the optimizer settings.
            The dictionaries can contain the following keys:
                - optimizer (str): Name of the optimizer.
                - parameters_group (list[str]): List of the names of the model parameters that are optimized by the optimizer.
                - learning_rate (float): Learning rate of the optimizer.
                - weight_decay (float): Weight decay of the optimizer.
                - lr_scheduler (str): Name of the learning rate scheduler.
                - warmup_steps (int): Number of warmup steps.
        max_steps (int):
            Maximum number of training epochs to perform. Defaults to 3.
        warmup_steps (int):
            Number of warmup steps. Defaults to 0.
        dtype (str):
            Data type to use. Defaults to "float32".
        **kwargs:
            Additional keyword arguments.

    Attributes:
        model (nn.Module):
            The model to wrap.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer used to tokenize the inputs.
        num_classes (int):
            Number of classes of the problem.
        id2label (dict):
            Mapping from class IDs to labels.
        label2id (dict):
            Mapping from labels to class IDs.
        optimizers_settings (list[dict]):
            List of dictionaries containing the optimizer settings.
        max_steps (int):
            Maximum number of training epochs to perform.
        training_step_index (int):
            Index of the training step.
        accuracy (torchmetrics.classification.Accuracy):
            Accuracy metric.
        precision (torchmetrics.classification.Precision):
            Precision metric.
        recall (torchmetrics.classification.Recall):
            Recall metric.
        f1_score (torchmetrics.classification.F1Score):
            F1-score metric.
        training_samples_count (int):
            Number of training samples.
        sum_training_epoch_loss (float):
            Sum of the training loss.
        training_stat_scores (ClassificationStats):
            Statistics of the training data.
        from_last_val_training_samples_count (int):
            Number of training samples from the last validation.
        sum_from_last_val_training_loss (float):
            Sum of the training loss from the last validation.
        from_last_val_training_stat_scores (ClassificationStats):
            Statistics of the training data from the last validation.
        validation_samples_count (int):
            Number of validation samples.
        sum_validation_epoch_loss (float):
            Sum of the validation loss.
        validation_stat_scores (ClassificationStats):
            Statistics of the validation data.
        test_samples_count (int):
            Number of test samples.
        sum_test_epoch_loss (float):
            Sum of the test loss.
        test_stat_scores (ClassificationStats):
            Statistics of the test data.
        model_dtype (str):
            Data type to use.
    """

    def __init__(
            self,
            model,
            tokenizer: transformers.PreTrainedTokenizer,
            num_classes: int,
            id2label: dict,
            label2id: dict,
            optimizers_settings: list[dict] = None,
            max_steps: int = 1,
            dtype: str = "float32",
            **kwargs
    ) -> None:
        super(ClassifierModelWrapper, self).__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.num_classes = num_classes
        self.id2label = id2label
        self.label2id = label2id

        self.optimizers_settings = optimizers_settings
        self.max_steps = max_steps

        self.training_step_index = 0

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.precision = torchmetrics.classification.Precision(
            task="multiclass",
            num_classes=num_classes
        )
        self.recall = torchmetrics.classification.Recall(
            task="multiclass",
            num_classes=num_classes
        )
        self.f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=num_classes
        )

        self.training_samples_count = 0
        self.sum_training_epoch_loss = 0
        self.training_stat_scores = ClassificationStats(num_classes=self.num_classes, average=None)

        self.from_last_val_training_samples_count = 0
        self.sum_from_last_val_training_loss = 0
        self.from_last_val_training_stat_scores = ClassificationStats(num_classes=self.num_classes, average=None)

        self.validation_samples_count = 0
        self.sum_validation_epoch_loss = 0
        self.validation_stat_scores = ClassificationStats(num_classes=self.num_classes, average=None)

        self.test_samples_count = 0
        self.sum_test_epoch_loss = 0
        self.test_stat_scores = ClassificationStats(num_classes=self.num_classes, average=None)

        self.model_dtype = dtype

    def configure_optimizers(
            self,
            **kwargs
    ) -> dict[str, torch.optim.Optimizer | str | Any]:
        """
        Configures the optimizer.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            list[dict[str, torch.optim.Optimizer | str | Any]]:
                List of dictionaries containing the optimizer and the learning rate scheduler.
        """

        if self.optimizers_settings is None or self.optimizers_settings == []:
            print("No optimizer settings provided, using default settings")
            self.optimizers_settings = [
                {
                    "optimizer": "AdamW",
                    "parameters_group": [
                        name
                        for name, param in self.model.named_parameters()
                    ],
                    "learning_rate": 1e-5,
                    "weight_decay": 0.01,
                    "lr_scheduler": "cosine_with_warmup",
                    "warmup_steps": 0,
                    "monitored_metric": "loss"
                }
            ]
        if not all(key in optimizer_settings for key in ["optimizer", "parameters_group", "learning_rate"] for optimizer_settings in self.optimizers_settings):
            raise ValueError("The optimizers' settings are not well defined, they should contain the keys 'optimizer', 'parameters_group' and 'learning_rate'")
        if not all(optimizer_settings["optimizer"].lower() in optimizers_mapping for optimizer_settings in self.optimizers_settings):
            raise ValueError(f"The following optimizers are not supported: {set(optimizer_settings['optimizer'] for optimizer_settings in self.optimizers_settings if optimizer_settings['optimizer'].lower() not in optimizers_mapping)}")
        if len(self.optimizers_settings) == 1 and len(self.optimizers_settings[0]["parameters_group"]) == 0:
            self.optimizers_settings[0]["parameters_group"] = [
                name
                for name, param in self.model.named_parameters()
            ]
        if len(self.optimizers_settings) > 1 and any(len(optimizer_settings["parameters_group"]) == 0 for optimizer_settings in self.optimizers_settings):
            raise ValueError("The parameters group of the optimizers' settings should not be empty")

        # Defining the optimizers
        optimizers = []
        for optimizer_settings in self.optimizers_settings:
            optimizer = optimizers_mapping[optimizer_settings["optimizer"].lower()](
                [param for name, param in self.model.named_parameters() if name in optimizer_settings["parameters_group"]],
                lr=optimizer_settings["learning_rate"],
                eps=1e-7 if self.model_dtype == "float16" else 1e-8
            )

            if "lr_scheduler" in optimizer_settings:
                new_optimizer = {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": transformers.get_cosine_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=optimizer_settings["warmup_steps"],
                            num_training_steps=self.max_steps,
                            num_cycles=0.5
                        ),
                        "monitor": optimizer_settings["monitored_metric"]
                    }
                }
            else:
                new_optimizer = {
                    "optimizer": optimizer
                }

            optimizers.append(new_optimizer)

        if len(optimizers) > 1:
            self.automatic_optimization = False

        return optimizers

    def forward(
            self,
            input_ids: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor):
                Input IDs.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Model outputs.
        """

        return self.model(
            input_ids,
            **kwargs
        ).logits

    def _common_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the common operations that training, validation and test step
        have to do.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
            torch.Tensor:
                Output computed by the model.
            torch.Tensor:
                Target labels for the current batch.
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = self.forward(
            input_ids,
            **{"attention_mask": attention_mask}
        )
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 2),
            labels.view(-1)
        )

        return loss, logits, labels

    def _regularization_step(
            self,
            loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a regularization step.

        Returns:
            torch.Tensor:
                Regularization loss.
        """

        # Computation of the regularization loss
        regularization_loss = self.model.get_training_penalization_loss(
                loss,
                self.training_step_index,
                self.device
        )

        # Logging
        logs_dicts = self.model.get_logging_info()
        for log_element in logs_dicts:
            self.log(**log_element)

        return regularization_loss

    def training_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a training step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        loss, logits, labels = self._common_step(batch, batch_idx)

        # Computation of the regularization loss
        if isinstance(self.model, RegularizedTrainingInterface):
            regularization_loss = self._regularization_step(loss)
        else:
            regularization_loss = 0.0

        # Computation of the total loss
        total_loss = loss + regularization_loss

        # Manual optimization in case of multiple optimizers
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                optimizer.zero_grad()

            self.manual_backward(total_loss, retain_graph=True)

            for optimizer in optimizers:
                optimizer.step()

        lr_schedulers = self.lr_schedulers()
        if isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
        else:
            lr_schedulers.step()

        # Logging
        self.log("loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.from_last_val_training_samples_count += logits.shape[0]
        self.sum_from_last_val_training_loss += loss.item() * logits.shape[0]
        self.from_last_val_training_stat_scores.update(logits.argmax(-1), labels)

        self.training_samples_count += logits.shape[0]
        self.sum_training_epoch_loss += loss.item() * logits.shape[0]
        self.training_stat_scores.update(logits.argmax(-1), labels)

        # Increasing the training step
        self.training_step_index += 1

        return total_loss

    def validation_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a validation step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        loss, logits, labels = self._common_step(batch, batch_idx)

        self.validation_samples_count += logits.shape[0]
        self.sum_validation_epoch_loss += loss.item() * logits.shape[0]
        self.validation_stat_scores.update(logits.argmax(-1), labels)

        return loss

    def test_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a test step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        loss, logits, labels = self._common_step(batch, batch_idx)

        self.test_samples_count += logits.shape[0]
        self.sum_test_epoch_loss += loss.item() * logits.shape[0]
        self.test_stat_scores.update(logits.argmax(-1), labels)

        return loss

    def predict_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int,
            **kwargs
    ) -> torch.Tensor:
        """
        Performs a prediction step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Prediction of the model computed for the current batch.
        """

        loss, logits, labels = self._common_step(batch, batch_idx)

        predicted_class_id = logits.argmax().item()

        return predicted_class_id

    def on_train_epoch_end(
            self
    ) -> None:
        """
        Performs operations at the end of a training epoch
        """

        self.log_dict(
            {
                "training_loss_epoch": self.sum_training_epoch_loss / self.training_samples_count,
                "training_accuracy_epoch": self.training_stat_scores.accuracy(),
                "training_precision_epoch": self.training_stat_scores.precision(),
                "training_recall_epoch": self.training_stat_scores.recall(),
                "training_f1_score_epoch": self.training_stat_scores.f1_score(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.training_samples_count = 0
        self.sum_training_epoch_loss = 0
        self.training_stat_scores.reset()

    def on_validation_epoch_end(
            self
    ) -> None:
        """
        Performs operations at the end of a validation epoch
        """

        print("##########################################################")
        if self.training_samples_count > 0:
            from_last_val_training_loss = self.sum_from_last_val_training_loss / self.training_samples_count
            from_last_val_training_accuracy = self.from_last_val_training_stat_scores.accuracy()
            from_last_val_training_precision = self.from_last_val_training_stat_scores.precision()
            from_last_val_training_recall = self.from_last_val_training_stat_scores.recall()
            from_last_val_training_f1_score = self.from_last_val_training_stat_scores.f1_score()

            print(f"Training Loss: {from_last_val_training_loss:.4f}")
            print(f"Training Accuracy: {from_last_val_training_accuracy:.4f}")
            print(f"Training Precision: {from_last_val_training_precision:.4f}")
            print(f"Training Recall: {from_last_val_training_recall:.4f}")
            print(f"Training F1-score: {from_last_val_training_f1_score:.4f}")

            self.log_dict(
                {
                    "from_last_val_training_loss": from_last_val_training_loss,
                    "from_last_val_training_accuracy": from_last_val_training_accuracy,
                    "from_last_val_training_precision": from_last_val_training_precision,
                    "from_last_val_training_recall": from_last_val_training_recall,
                    "from_last_val_training_f1_score": from_last_val_training_f1_score,
                },
                on_step=False,
                on_epoch=True
            )

        else:
            print("No data about the training")

        print("----------------------------------------------------------")

        if self.validation_samples_count > 0:
            validation_loss = self.sum_validation_epoch_loss / self.validation_samples_count
            validation_accuracy = self.validation_stat_scores.accuracy()
            validation_precision = self.validation_stat_scores.precision()
            validation_recall = self.validation_stat_scores.recall()
            validation_f1_score = self.validation_stat_scores.f1_score()

            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print(f"Validation Precision: {validation_precision:.4f}")
            print(f"Validation Recall: {validation_recall:.4f}")
            print(f"Validation F1-score: {validation_f1_score:.4f}")

            self.log_dict(
                {
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_precision": validation_precision,
                    "validation_recall": validation_recall,
                    "validation_f1_score": validation_f1_score,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )

        print("##########################################################")

        self.from_last_val_training_samples_count = 0
        self.sum_from_last_val_training_loss = 0
        self.from_last_val_training_stat_scores.reset()

        self.validation_samples_count = 0
        self.sum_validation_epoch_loss = 0
        self.validation_stat_scores.reset()

    def on_test_epoch_end(
            self
    ) -> None:
        """
        Performs operations at the end of a test epoch
        """

        if self.test_samples_count > 0:
            test_loss = self.sum_test_epoch_loss / self.test_samples_count
            test_accuracy = self.test_stat_scores.accuracy()
            test_precision = self.test_stat_scores.precision()
            test_recall = self.test_stat_scores.recall()
            test_f1_score = self.test_stat_scores.f1_score()

            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test F1-score: {test_f1_score:.4f}")

            self.log_dict(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1_score": test_f1_score,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )

    def predict(
            self,
            text: str,
            verbose: bool
    ) -> str:
        """
        Predicts the class of the input text.

        Args:
            text (str):
                Input text to classify.
            verbose (bool):
                Whether to print the logits and the predicted class.

        Returns:
            str:
                Predicted label.
        """

        x = self.tokenizer(
            text,
            return_tensors="pt"
        )

        x = x.to(self.model.device)

        with torch.no_grad():
            logits = self.model(**x)

        predicted_class_id = logits.argmax().item()
        predicted_label = self.id2label[predicted_class_id]

        if verbose:
            print(f"Logits: {logits}")
            print(f"Predicted class: {predicted_class_id}")
            print(f"Predicted label: {predicted_label}")

        return predicted_class_id
