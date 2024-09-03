from __future__ import annotations

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


class RandomDataset(SentimentAnalysisDataset):
    """
    Random dataset for sentiment analysis.

    Args:
        size (int):
            Size of the dataset.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        size (int):
            Size of the dataset.
        hidden_size (int):
            Hidden size of the dataset.
        dataset (datasets.Dataset):
            Random embeddings.
    """

    def __init__(
            self,
            size: int,
            hidden_size: int,
            max_length: int = 512,
            **kwargs
    ) -> None:
        super().__init__(
            "random",
            None
        )

        self.size = size
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.dataset = torch.randn(self.size, self.max_length, self.hidden_size), torch.randint(0, 2, (self.size,))

    def __len__(
            self
    ) -> int:
        """
        Returns the length of the dataset.
        """

        return self.size

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

        sample = self.dataset[0][idx]
        label = self.dataset[1][idx]

        # Converting input_ids, attention_mask, and labels to tensors
        input_ids = torch.tensor(sample["input_ids"])
        attention_mask = torch.tensor(sample["attention_mask"])
        label = torch.tensor(sample["label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

        return self.dataset[0][idx], self.dataset[1][idx]


class RandomDataModule(pl.LightningDataModule):
    """
    DataModule for the random dataset.

    Args:
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for data loading.
        size (int):
            Size of the dataset.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for loading the data.
        size (int):
            Size of the dataset.
        train (RandomDataset):
            Training dataset.
        validation (RandomDataset):
            Validation dataset.
        test (RandomDataset):
            Test dataset.
    """

    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            size: int,
            hidden_size: int,
            split: tuple[float, float, float],
            **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.hidden_size = hidden_size
        self.split = split

        self.train = None
        self.validation = None
        self.test = None

    def prepare_data(
            self,
            **kwargs
    ):
        """
        Downloads the data. Runs on a single GPU.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        pass

    def setup(
            self,
            stage: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Preprocesses data. Can run on multiple GPUs.

        Args:
            stage (Optional[str]):
                Stage of the experiment.
            kwargs:
                Additional keyword arguments.
        """

        self.train = RandomDataset(self.size)
        self.validation = RandomDataset(self.size)
        self.test = RandomDataset(self.size)

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


class IMDBDataset(SentimentAnalysisDataset):
    """
    Dataset class to use IMDB dataset with Pytorch and Pytorch Lightning.

    Args:
        raw_dataset (datasets.Dataset):
            Raw dataset.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.
        dataset (datasets.Dataset):
            Tokenized dataset.
    """

    def __init__(
            self,
            raw_dataset: datasets.Dataset,
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 512,
            **kwargs
    ) -> None:
        super().__init__(
            "stanfordnlp/imdb",
            tokenizer
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        tokenized_dataset = raw_dataset.map(self.preprocess_function, batched=True)

        self.dataset = tokenized_dataset

    def preprocess_function(
            self,
            examples,
            **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Preprocess function to use on the dataset.
        It tokenizes the text and pads it to the maximum length.

        Args:
            examples:
                Examples to preprocess.
            **kwargs:
                Additional keyword arguments.
        """

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


class IMDBDataModule(pl.LightningDataModule):
    """
    DataModule for the IMDB dataset.

    Args:
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for data loading.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets.
        seed (int):
            Seed for the random number generator.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for loading the data.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets.
        seed (int):
            Seed for the random number generator.
        train (IMDBDataset):
            Training dataset.
        validation (IMDBDataset):
            Validation dataset.
        test (IMDBDataset):
            Test dataset.
    """

    def __init__(
            self,
            batch_size,
            num_workers,
            tokenizer,
            max_len: int,
            split: tuple[float, float, float],
            seed: int = 42,
            **kwargs
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
            self,
            **kwargs
    ):
        """
        Downloads the data. Runs on a single GPU.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        load_dataset(
            "stanfordnlp/imdb",
            #data_dir=self.data_dir,
        )

    def setup(
            self,
            stage: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Preprocesses data. Can run on multiple GPUs.


        Args:
            stage (Optional[str]):
                Stage of the experiment.
            kwargs:
                Additional keyword arguments.
        """

        raw_dataset = load_dataset(
            "stanfordnlp/imdb"
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

        self.train = IMDBDataset(
            second_split_raw_dataset["train"],
            self.tokenizer,
            self.max_len
        )
        self.validation = IMDBDataset(
            second_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )
        self.test = IMDBDataset(
            first_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )

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
