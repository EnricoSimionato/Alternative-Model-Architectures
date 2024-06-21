from __future__ import annotations

from typing import Any

import torch
import torchmetrics

import pytorch_lightning as pl

import transformers

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
        kfc_training (bool):
            Whether to perform KFC training. Defaults to False.
        initial_regularization_weight (float):
            Initial regularization weight. Defaults to 0.01.
        max_regularization_weight (torch.Tensor):
            Maximum regularization weight. Defaults to 10000.0.
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
        learning_rates (list):
            Learning rate.
        max_steps (int):
            Maximum number of training epochs to perform.
        warmup_steps (int):
            Number of warmup steps.
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
            Sum of the training loss in the current epoch.
        training_stat_scores (ClassificationStats):
            Statistics of the training data.
        from_last_val_training_samples_count (int):
            Number of training samples from the last validation epoch.
        sum_from_last_val_training_loss (float):
            Sum of the training loss from the last validation epoch.
        from_last_val_training_stat_scores (ClassificationStats):
            Statistics of the training data from the last validation epoch.
        validation_samples_count (int):
            Number of validation samples.
        sum_validation_epoch_loss (float):
            Sum of the validation loss in the current epoch.
        validation_stat_scores (ClassificationStats):
            Statistics of the validation data.
        test_samples_count (int):
            Number of test samples.
        sum_test_epoch_loss (float):
            Sum of the test loss in the current epoch.
        test_stat_scores (ClassificationStats):
            Statistics of the test data.
        model_dtype (str):
            Data type to use.
    """

    weights_to_exclude = [
        "lora",
        "vera"
    ]

    # TODO log also the norm of the adapters
    # TODO Increase the regularization

    def __init__(
            self,
            model,
            tokenizer: transformers.PreTrainedTokenizer,
            num_classes: int,
            id2label: dict,
            label2id: dict,
            optimizers_settings: list[dict] = None,
            max_steps: int = 1,
            kfc_training: bool = False,
            initial_regularization_weight: float = 0.01,
            max_regularization_weight: float = 10.0,
            start_step_regularization: int = 0,
            steps_regularization_weight_resets: int = 1000,
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

        self.kfc_training = kfc_training
        self.initial_regularization_weight = initial_regularization_weight
        self.fixed_regularization_weight = None
        self.adaptive_regularization_weight = torch.tensor(
            initial_regularization_weight,
            requires_grad=False
        )
        self.max_regularization_weight = torch.tensor(
            max_regularization_weight,
            requires_grad=False
        )
        self.start_step_regularization = start_step_regularization
        self.steps_regularization_weight_resets = steps_regularization_weight_resets

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
            self.optimizers_settings = [
                {
                    "optimizer": "adamw",
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

        optimizers = []
        for optimizer_settings in self.optimizers_settings:
            optimizer = optimizers_mapping[optimizer_settings["optimizer"].lower()](
                [param for name, param in self.model.named_parameters() if name in optimizer_settings["parameters_group"]],
                lr=optimizer_settings["learning_rate"],
                eps=1e-7 if self.model_dtype == "float16" else 1e-8
            )

            if "lr_scheduler" in optimizer_settings:
                # TODO: Add the possibility to use different learning rate schedulers
                # TODO: Pass to optimizer and lr_scheduler a dictionary of parameters
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

    def get_unweighted_penalization(
            self,
            layers_to_penalize: list[str]
    ) -> torch.Tensor:
        """
        Computes the unweighted penalization term as the L1 norm of the model weights.

        Args:
            layers_to_penalize (list[str]):
                List of the names of the model parameters that are penalized.

        Returns:
            torch.Tensor:
                Unweighted penalization term.
        """

        original_params = []
        for name, param in self.model.named_parameters():
            if name in layers_to_penalize:
                original_params.append(param)

        sum_l1_norms = torch.tensor(0.0, device=self.device)
        for i, param in enumerate(original_params):
            sum_l1_norms += torch.sum(torch.abs(param.flatten()))

        return sum_l1_norms

    def get_weighted_penalization(
            self,
            penalization: torch.Tensor,
            loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the weighted penalization term.

        Args:
            penalization (torch.Tensor):
                Unweighted penalization term.
            loss (torch.Tensor):
                Loss of the model computed for the current batch.

        Returns:
            torch.Tensor:
                Weighted penalization term.
        """

        if self.fixed_regularization_weight is None:
            self.fixed_regularization_weight = torch.tensor(
                2 * (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
        elif (self.steps_regularization_weight_resets > 0 and
              self.training_step_index % self.steps_regularization_weight_resets == 0):
            self.fixed_regularization_weight = torch.tensor(
                2 * (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
            self.adaptive_regularization_weight = torch.tensor(
                self.initial_regularization_weight,
                requires_grad=False
            )
            print("Fixed regularization weight reset to", self.fixed_regularization_weight.item(), "and adaptive regularization weight reset to", self.adaptive_regularization_weight.item())

        self.log(
            "fixed_regularization_weight",
            self.fixed_regularization_weight,
            on_step=True,
            on_epoch=False
        )

        return penalization * self.fixed_regularization_weight

    def regularization_scheduler_step(
            self
    ):
        """
        Updates the regularization weight.
        """

        k = torch.sqrt(torch.tensor(
            1.1,
            requires_grad=False
        )).to(self.adaptive_regularization_weight.device)
        self.adaptive_regularization_weight = torch.min(
            self.max_regularization_weight.to(self.adaptive_regularization_weight.device),
            self.adaptive_regularization_weight * k
        )

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

        if self.kfc_training and self.training_step_index >= self.start_step_regularization:
            self.log(
                "task_loss",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

            self.log(
                "adaptive_regularization_weight",
                self.adaptive_regularization_weight,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

            unweighted_penalization = self.get_unweighted_penalization(
                [
                    name
                    for name, param in self.model.named_parameters()
                    if not any(substring in name for substring in ClassifierModelWrapper.weights_to_exclude)
                ]
            )
            self.log(
                "unweighted_penalization",
                unweighted_penalization,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

            self.log(
                "norm of non-regularized weights",
                self.get_unweighted_penalization(
                    [
                        name
                        for name, param in self.model.named_parameters()
                        if any(substring in name for substring in ClassifierModelWrapper.weights_to_exclude)
                    ]
                ),
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

            weighted_penalization = self.get_weighted_penalization(unweighted_penalization, loss)
            self.log(
                "weighted_penalization",
                weighted_penalization,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

            weighted_penalization = weighted_penalization.to(loss.device)
            self.adaptive_regularization_weight = self.adaptive_regularization_weight.to(loss.device)

            loss = loss + self.adaptive_regularization_weight * weighted_penalization

            self.regularization_scheduler_step()

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        self.from_last_val_training_samples_count += logits.shape[0]
        self.sum_from_last_val_training_loss += loss.item() * logits.shape[0]
        self.from_last_val_training_stat_scores.update(logits.argmax(-1), labels)

        self.training_samples_count += logits.shape[0]
        self.sum_training_epoch_loss += loss.item() * logits.shape[0]
        self.training_stat_scores.update(logits.argmax(-1), labels)

        print(self.training_step_index)

        self.training_step_index += 1

        return loss

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
            batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a prediction step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Prediction of the model computed for the current batch.
        """

        loss, logits, labels = self._common_step(batch, batch_idx)

        predicted_class_id = logits.argmax().item()

        return predicted_class_id

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
