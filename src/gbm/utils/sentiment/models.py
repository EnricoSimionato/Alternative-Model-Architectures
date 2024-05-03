import numpy as np

import torch
import torchmetrics

import pytorch_lightning as pl

import transformers
from transformers.optimization import AdamW

from torch.utils.data import Dataset, DataLoader


class ClassifierModelWrapper(pl.LightningModule):
    """
    Wrapper to train a classifier model in Pytorch Lightning.


    Args:
        model (Any): The model to wrap.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer object.
        train_data (Union[Dataset, DataLoader]):
            Training data.
        val_data (Union[Dataset, DataLoader]):
            Validation data.
        test_data (Union[Dataset, DataLoader]):
            Test data.
        num_classes (int):
            Number of classes of the problem.
        id2label (dict):
            Mapping from class IDs to labels.
        label2id (dict):
            Mapping from labels to class IDs.
        batch_size (int):
            Batch size. Defaults to 4.
        learning_rate (float):
            Learning rate. Defaults to 1e-5.

    Attributes:

    """

    def __init__(
            self,
            model,
            tokenizer: transformers.PreTrainedTokenizer,
            train_data: Dataset,
            val_data: Dataset,
            test_data: Dataset,
            num_classes: int,
            id2label: dict,
            label2id: dict,
            batch_size: int = 16,
            learning_rate: float = 1e-5
    ) -> None:
        super(ClassifierModelWrapper, self).__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.num_classes = num_classes
        self.id2label = id2label
        self.label2id = label2id

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=num_classes
        )

        self.training_samples_count = 0
        self.from_last_val_training_loss = 0
        self.from_last_val_training_accuracy = 0
        self.from_last_val_training_f1_score = 0

        self.validation_samples_count = 0
        self.sum_epoch_validation_loss = 0
        self.sum_epoch_validation_accuracy = 0
        self.sum_epoch_validation_f1_score = 0

    def configure_optimizers(
            self
    ) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer:
                Optimizer.
        """

        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate
        )

        return optimizer

    def train_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the training DataLoader.

        Returns:
            DataLoader:
                Training DataLoader.
        """

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True
        )

    def val_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader:
                Validation DataLoader.
        """

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size * 2,
            num_workers=2
        )

    def test_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the test DataLoader.

        Returns:
            DataLoader:
                Test DataLoader.
        """

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size * 2,
            num_workers=2
        )

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
        accuracy = self.accuracy(logits, labels)
        f1_score = self.f1_score(logits, labels)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log_dict(
            {
                "training_loss": loss,
                "trianing_accuracy": accuracy,
                "training_f1_score": f1_score
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.training_samples_count += len(logits)

        self.from_last_val_training_loss += loss * len(logits)
        self.from_last_val_training_accuracy += accuracy * len(logits)
        self.from_last_val_training_f1_score += f1_score * len(logits)

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
        accuracy = self.accuracy(logits, labels)
        f1_score = self.f1_score(logits, labels)

        self.log_dict(
            {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
                "validation_f1_score": f1_score
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.validation_samples_count += len(logits)

        self.sum_epoch_validation_loss += loss * len(logits)
        self.sum_epoch_validation_accuracy += accuracy * len(logits)
        self.sum_epoch_validation_f1_score += f1_score * len(logits)

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
        accuracy = self.accuracy(logits, labels)
        f1_score = self.f1_score(logits, labels)

        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_f1_score": f1_score
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

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
    ) -> torch.Tensor:
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

    def on_validation_epoch_end(
            self
    ) -> None:
        """
        Performs operations at the end of a validation epoch
        """

        print("##########################################################")
        if self.training_samples_count > 0:
            avg_from_last_val_training_loss = self.from_last_val_training_loss / self.training_samples_count
            avg_from_last_val_training_accuracy = self.from_last_val_training_accuracy / self.training_samples_count
            avg_from_last_val_training_f1_score = self.from_last_val_training_f1_score / self.training_samples_count

            print(f"Training Loss: {avg_from_last_val_training_loss:.4f}")
            print(f"Training Accuracy: {avg_from_last_val_training_accuracy:.4f}")
            print(f"Training F1 Score: {avg_from_last_val_training_f1_score:.4f}")

            self.log_dict(
            {
                "from_last_val_training_loss": avg_from_last_val_training_loss,
                "from_last_val_training_accuracy": avg_from_last_val_training_accuracy,
                "from_last_val_training_f1_score": avg_from_last_val_training_f1_score
            },
            on_step=False,
            on_epoch=True
        )
        else:
            print("No data about the training")


        print("----------------------------------------------------------")

        if self.validation_samples_count > 0:
            validation_loss = self.sum_epoch_validation_loss / self.validation_samples_count
            validation_accuracy = self.sum_epoch_validation_accuracy / self.validation_samples_count
            validation_f1_score = self.sum_epoch_validation_f1_score / self.validation_samples_count

            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print(f"Validation F1 Score: {validation_f1_score:.4f}")

        print("##########################################################")

        self.training_samples_count = 0
        self.from_last_val_training_loss = 0
        self.from_last_val_training_accuracy = 0
        self.from_last_val_training_f1_score = 0

        self.validation_samples_count = 0
        self.sum_epoch_validation_loss = 0
        self.sum_epoch_validation_accuracy = 0
        self.sum_epoch_validation_f1_score = 0

    def on_test_epoch_end(
            self
    ) -> None:
        """
        Performs operations at the end of a test epoch
        """
        pass

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
