from __future__ import annotations

from abc import ABC, abstractmethod

import datasets
import torch

from torch.utils.data import Dataset

from transformers import AutoTokenizer

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


class ConversationDataset(ABC, Dataset):
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


class IMDBDatasetDict:

    def __init__(
            self,
            tokenizer: AutoTokenizer,
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
        merged_raw_dataset = concatenate_datasets([raw_dataset[key] for key in raw_dataset.keys()])

        first_split_raw_dataset = merged_raw_dataset.train_test_split(
            test_size=split[2]
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=split[1]/((1-split[2]) if split[2] != 1 else 1)
        )

        self.train = IMDBDataset(second_split_raw_dataset["train"], tokenizer, max_len)
        self.validation = IMDBDataset(second_split_raw_dataset["test"], tokenizer, max_len)
        self.test = IMDBDataset(first_split_raw_dataset["test"], tokenizer, max_len)


class IMDBDataset(SentimentAnalysisDataset):
    """
    IMDB dataset.
    """

    def __init__(
            self,
            raw_dataset: datasets.Dataset,
            tokenizer,
            max_length: int = 512
    ) -> None:
        """
        Initializes the dataset loading it from HuggingFace.
        """

        super().__init__("imdb", tokenizer)

        self.tokenizer = tokenizer
        self.max_length = max_length

        tokenized_dataset = raw_dataset.map(self.preprocess_function, batched=True)

        self.dataset = tokenized_dataset

    def preprocess_function(
            self,
            examples
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized_example = self.tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.
        """

        return self.dataset[idx]


class OpenassistantGuanacoDatasetDict:
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            max_len: int,
            split: tuple[float, float, float]
    ) -> None:
        """
        Initializes the dataset loading it from .
        """

        if len(split) != 3:
            raise ValueError(
                "The split must have three elements (train, validation, test)."
            )
        if sum(split) != 1:
            raise ValueError(
                "The sum of the split elements must be equal to 1."
            )

        raw_dataset = load_dataset("openassistant-guanaco")
        merged_raw_dataset = concatenate_datasets([raw_dataset[key] for key in raw_dataset.keys()])

        first_split_raw_dataset = merged_raw_dataset.train_test_split(
            test_size=split[2]
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=split[1]/((1-split[2]) if split[2] != 1 else 1)
        )

        self.train = OpenassistantGuanacoDataset(second_split_raw_dataset["train"], tokenizer, max_len)
        self.validation = OpenassistantGuanacoDataset(second_split_raw_dataset["test"], tokenizer, max_len)
        self.test = OpenassistantGuanacoDataset(first_split_raw_dataset["test"], tokenizer, max_len)


class OpenassistantGuanacoDataset(ConversationDataset):
    """
    Openassistant-guanaco dataset.
    """

    def __init__(
            self,
            raw_dataset: datasets.Dataset,
            tokenizer,
            max_length: int = 512
    ) -> None:
        """
        Initializes the dataset loading it from HuggingFace.
        """

        super().__init__("imdb", tokenizer)

        self.tokenizer = tokenizer
        self.max_length = max_length

        tokenized_dataset = raw_dataset.map(self.preprocess_function, batched=True)

        self.input_ids = torch.tensor(tokenized_dataset["input_ids"])
        self.attention_masks = torch.tensor(tokenized_dataset["attention_mask"])
        self.labels = torch.tensor(tokenized_dataset["label"])
        print(tokenized_dataset)

    def preprocess_function(
            self,
            examples
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized_example = self.tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        return tokenized_example

    def __len__(
            self
    ) -> int:
        """
        Returns the length of the dataset.
        """

        return len(self.input_ids)

    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.
        """

        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]


class LightningDataset(Dataset):
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


if  __name__ == "__main__":
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = IMDBDatasetDict(
        bert_tokenizer,
        512,
        (0.8, 0.1, 0.1)
    )

    for i in range(10):
        print(dataset.train[i])
        print(dataset.validation[i])
        print(dataset.test[i])
        print()
