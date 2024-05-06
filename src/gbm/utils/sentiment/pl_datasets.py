from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import transformers

import datasets

from datasets import load_dataset, concatenate_datasets


class SentimentAnalysisDataset(ABC, Dataset):
    """
    Dataset for sentiment analysis.
    """

    def __init__(
            self,
            dataset_id: str,
            tokenizer
    ) -> None:
        """
        Initializes the SentimentAnalysisDataset.
        """

        self.dataset_id = dataset_id
        self.tokenizer = tokenizer

    @abstractmethod
    def __len__(
            self
    ) -> int:
        """
        Returns the length of the dataset.
        """

    @abstractmethod
    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.
        """


class IMDBDataset(SentimentAnalysisDataset):
    """
    IMDB dataset.
    """

    def __init__(
            self,
            raw_dataset: datasets.Dataset,
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 512
    ) -> None:
        super().__init__("imdb", tokenizer)

        self.tokenizer = tokenizer
        self.max_length = max_length

        tokenized_dataset = raw_dataset.map(self.preprocess_function, batched=True)

        self.dataset = tokenized_dataset

    def preprocess_function(
            self,
            examples
    ) -> dict[str, torch.Tensor]:
        tokenized_example = self.tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",
        )

        return tokenized_example

    def __len__(
            self
    ) -> int:
        """
        Returns the length of the dataset.
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
        """

        sample = self.dataset[idx]

        # Converting input_ids, attention_mask, and labels to tensors
        input_ids = torch.tensor(sample["input_ids"])
        attention_mask = torch.tensor(sample["attention_mask"])
        label = torch.tensor(sample["label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }


class IMDBDatasetDict:

    def __init__(
            self,
            tokenizer,
            max_len: int,
            split: tuple[float, float, float]
    ) -> None:
        """
        Initializes the dataset loading it from HuggingFace.
        """

        if len(split) != 3:
            raise ValueError(
                "The split must have three elements (train, validation, test)."
            )
        if sum(split) != 1:
            raise ValueError(
                "The sum of the split elements must be equal to 1."
            )

        raw_dataset = load_dataset("imdb")

        concatenated_raw_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])

        first_split_raw_dataset = concatenated_raw_dataset.train_test_split(
            test_size=split[2]
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=split[1]/((1-split[2]) if split[2] != 1 else 1)
        )

        self.train = IMDBDataset(second_split_raw_dataset["train"], tokenizer, max_len)
        self.validation = IMDBDataset(second_split_raw_dataset["test"], tokenizer, max_len)
        self.test = IMDBDataset(first_split_raw_dataset["test"], tokenizer, max_len)


class IMDBDataModule(pl.LightningDataModule):
    """

    """

    def __init__(
            self,
            data_dir,
            batch_size,
            num_workers,
            tokenizer,
            max_len: int,
            split: tuple[float, float, float],
            seed: int = 42
    ):
        super().__init__()
        if len(split) != 3:
            raise ValueError(
                "The split must have three elements (train, validation, test)."
            )
        if sum(split) != 1:
            raise ValueError(
                "The sum of the split elements must be equal to 1."
            )

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.seed = seed

        self.train = None
        self.validation = None
        self.test = None

    def prepare_data(
            self
    ):
        """
        Downloads the data. Run on a single GPU.
        """

        load_dataset(
            "imdb",
            data_dir=self.data_dir,
        )

    def setup(
            self,
            stage: Optional[str] = None
    ):
        """
        Preprocesses data. Run on multiple GPUs.
        """

        raw_dataset = load_dataset(
            "imdb",
            data_dir=self.data_dir,
        )

        concatenated_raw_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])

        first_split_raw_dataset = concatenated_raw_dataset.train_test_split(
            test_size=self.split[2],
            seed=self.seed
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=self.split[1]/((1-self.split[2]) if self.split[2] != 1 else 1),
            seed=self.seed
        )

        self.train = IMDBDataset(second_split_raw_dataset["train"], self.tokenizer, self.max_len)
        self.validation = IMDBDataset(second_split_raw_dataset["test"], self.tokenizer, self.max_len)
        self.test = IMDBDataset(first_split_raw_dataset["test"], self.tokenizer, self.max_len)

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
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
            self.validation,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
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
            self.test,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
        )


if __name__ == "__main__":

    from transformers import AutoTokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = IMDBDataModule(
        os.getcwd(),
        32,
        2,
        bert_tokenizer,
        512,
        (0.8, 0.1, 0.1),
        seed=42
    )

    print(dataset.train)
    print()