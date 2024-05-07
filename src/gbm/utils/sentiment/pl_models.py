import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics

import pytorch_lightning as pl

import transformers

from gbm.utils.sentiment.pl_metrics import ClassificationStats


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
        learning_rate (float):
            Learning rate. Defaults to 1e-5.
        max_epochs (int):
            Maximum number of training epochs to perform. Defaults to 3.

    Attributes:
        model (Any):
            The model to wrap.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer object.
        num_classes (int):
            Number of classes of the problem.
        id2label (dict):
            Mapping from class IDs to labels.
        label2id (dict):
            Mapping from labels to class IDs.
        learning_rate (float):
            Learning rate.
        max_epochs (int):
            Maximum number of training epochs to perform.
        accuracy (torchmetrics.classification.Accuracy):
            Accuracy metric.
        f1_score (torchmetrics.classification.F1Score):
            F1 score metric.
        training_samples_count (int):
            Number of training samples.
    """

    def __init__(
            self,
            model,
            tokenizer: transformers.PreTrainedTokenizer,
            num_classes: int,
            id2label: dict,
            label2id: dict,
            learning_rate: float = 1e-5,
            max_epochs: int = 3,
    ) -> None:
        super(ClassifierModelWrapper, self).__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.num_classes = num_classes
        self.id2label = id2label
        self.label2id = label2id

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.accuracy = torchmetrics.classification.Accuracy(
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

    def configure_optimizers(
            self,
            **kwargs
    ) -> dict:
        """
        Configures the optimizer.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.optim.Optimizer:
                Optimizer.
        """

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate
        )

        learning_rate_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs
        )
        monitored_metrics = "loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": learning_rate_scheduler,
            "monitor": monitored_metrics
        }

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
