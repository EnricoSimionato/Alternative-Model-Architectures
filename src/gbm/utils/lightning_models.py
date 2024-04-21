import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # Extract input_ids, attention_mask, and labels from the batch
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences dynamically to the maximum length within each batch
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Convert labels to tensor
    padded_labels = torch.tensor(labels)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks,
        "labels": padded_labels
    }


class LightningDataset(TorchDataset):
    """
    Dataset to use in the training with Pytorch Lightning.
    """

    def __init__(
            self,
            dataset: Dataset
    ) -> None:
        """
        Initializes the Dataset to use in the training with Pytorch Lightning.

        Args:
            dataset (Dataset):
                The dataset to be wrapped.
        """

        self.dataset = dataset

    def __len__(
            self
    ) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int:
                Length of the dataset.
        """

        return len(self.dataset)

    def __getitem__(
            self,
            idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            dict[str, torch.Tensor]:
                Dictionary containing the tokenized inputs.
        """

        sample = self.dataset[idx]

        # Converting input_ids, attention_mask, and labels to tensors
        input_ids = torch.tensor(sample["input_ids"])
        attention_mask = torch.tensor(sample["attention_mask"])
        labels = torch.tensor(sample["label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class BinaryClassifier(pl.LightningModule):
    """
    Wrapper to train the model in Pytorch Lightning.
    """

    def __init__(
            self,
            model,
            tokenizer: AutoTokenizer,
            train_data: LightningDataset,
            val_data: LightningDataset,
            test_data: LightningDataset,
            batch_size: int = 16,
            learning_rate: float = 1e-5,
    ) -> None:
        """
        Initializes the LightningModelWrapper.

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
            batch_size (int):
                Batch size. Defaults to 4.
            learning_rate (float):
                Learning rate. Defaults to 1e-5.
        """

        super(BinaryClassifier, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer:
                Optimizer.
        """

        # Defining the optimizer to use
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

        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader:
                Validation DataLoader.
        """

        return DataLoader(self.val_data, batch_size=self.batch_size * 2, collate_fn=collate_fn)

    def test_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the test DataLoader.

        Returns:
            DataLoader:
                Test DataLoader.
        """

        return DataLoader(self.test_data, batch_size=self.batch_size * 2, collate_fn=collate_fn)

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
        labels = batch["labels"]

        logits = self.forward(input_ids, **{"attention_mask": attention_mask})

        loss = torch.nn.functional.cross_entropy(logits.view(-1, 2), labels.view(-1))

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
        labels = batch["labels"]

        logits = self.forward(input_ids, **{"attention_mask": attention_mask})

        loss = torch.nn.functional.cross_entropy(logits.view(-1, 2), labels.view(-1))

        self.validation_step_losses_sum += loss.detach().cpu().numpy()
        self.validation_step_losses_count += 1

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
        labels = batch["labels"]

        logits = self.forward(input_ids, **{"attention_mask": attention_mask})

        loss = torch.nn.functional.cross_entropy(logits.view(-1, 2), labels.view(-1))

        self.test_step_losses_sum += loss.detach().cpu().numpy()
        self.test_step_losses_count += 1

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
