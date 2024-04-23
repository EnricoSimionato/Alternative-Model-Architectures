import numpy as np

import torch

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import AutoTokenizer
from transformers.optimization import AdamW

from gbm.utils.lightning_datasets import ConversationDataset


class ClassifierModelWrapper(pl.LightningModule):
    """
    Wrapper to train a classifier model in Pytorch Lightning.
    """

    def __init__(
            self,
            model,
            tokenizer: AutoTokenizer,
            train_data: ConversationDataset,
            val_data: ConversationDataset,
            test_data: ConversationDataset,
            id2label: dict,
            label2id: dict,
            batch_size: int = 16,
            learning_rate: float = 1e-5
    ) -> None:
        """
        Initializes the ClassifierModelWrapper.

        Args:
            model (Any): The model to wrap.
            tokenizer (transformers.AutoTokenizer):
                Tokenizer object.
            train_data (Union[Dataset, DataLoader]):
                Training data.
            val_data (Union[Dataset, DataLoader]):
                Validation data.
            test_data (Union[Dataset, DataLoader]):
                Test data.
            id2label (dict):
                Mapping from class IDs to labels.
            label2id (dict):
                Mapping from labels to class IDs.
            batch_size (int):
                Batch size. Defaults to 4.
            learning_rate (float):
                Learning rate. Defaults to 1e-5.
        """

        super(ClassifierModelWrapper, self).__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.id2label = id2label
        self.label2id = label2id

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.training_step_index = 0
        self.loss_history = {
            "train": [],
            "validation": [],
            "test": []
        }
        self.training_step_losses_sum = 0
        self.training_step_losses_count = 0
        self.validation_step_losses_sum = 0
        self.validation_step_losses_count = 0
        self.test_step_losses_sum = 0
        self.test_step_losses_count = 0

    def configure_optimizers(
            self
    ) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer:
                Optimizer.
        """

        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

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

        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)#, collate_fn=collate_fn)

    def val_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader:
                Validation DataLoader.
        """

        return DataLoader(self.val_data, batch_size=self.batch_size * 2)#, collate_fn=collate_fn)

    def test_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the test DataLoader.

        Returns:
            DataLoader:
                Test DataLoader.
        """

        return DataLoader(self.test_data, batch_size=self.batch_size * 2)#, collate_fn=collate_fn)

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

        return self.model(input_ids, **kwargs).logits

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

        self.training_step_losses_sum += loss.detach().cpu().numpy()
        self.training_step_losses_count += 1

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

        self.validation_step_losses_sum += loss.detach().cpu().numpy()
        self.validation_step_losses_count += 1

        self.log("val_loss", loss)

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

        self.test_step_losses_sum += loss.detach().cpu().numpy()
        self.test_step_losses_count += 1

        self.log("test_loss", loss)

        return loss

    def on_validation_epoch_end(
            self
    ) -> None:
        """
        Computes and stores the average training loss on the samples considered from the previous
        validation check to the current one and the average loss on the validation set.
        """

        if self.training_step_losses_count > 0:
            avg_train_loss = self.training_step_losses_sum / self.training_step_losses_count
            self.loss_history["train"].append(avg_train_loss)
        else:
            avg_train_loss = np.nan

        self.training_step_losses_sum = 0
        self.training_step_losses_count = 0

        if self.validation_step_losses_count > 0:
            avg_val_loss = self.validation_step_losses_sum / self.validation_step_losses_count
            self.loss_history["validation"].append(avg_val_loss)
        else:
            avg_val_loss = np.nan

        self.validation_step_losses_sum = 0
        self.validation_step_losses_count = 0

        print("----------------------------------------------------------")
        if np.isnan(avg_train_loss):
            print("Number of training steps equal to 0")
        print(f'Training loss: {avg_train_loss}')
        if np.isnan(avg_val_loss):
            print("Number of validation steps equal to 0")
        print(f'Validation loss: {avg_val_loss}')
        print("----------------------------------------------------------")

    def on_test_epoch_end(
            self
    ) -> None:
        """
        Computes and stores the average validation on the test set.
        """

        if self.test_step_losses_count > 0:
            avg_test_loss = self.test_step_losses_sum / self.test_step_losses_count
            self.loss_history["test"].append(avg_test_loss)
        else:
            avg_test_loss = np.nan

        self.test_step_losses_sum = 0
        self.test_step_losses_count = 0

        print("----------------------------------------------------------")
        if np.isnan(avg_test_loss):
            print("Number of test steps equal to 0")
        print(f'Test loss: {avg_test_loss}')
        print("----------------------------------------------------------")

    def predict(
            self,
            text: str
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

        print(f"Logits: {logits}")
        print(f"Predicted class: {predicted_class_id}")
        print(f"Predicted label: {predicted_label}")

        return predicted_label
