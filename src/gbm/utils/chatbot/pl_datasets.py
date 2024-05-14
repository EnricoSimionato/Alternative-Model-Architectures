from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import transformers

import datasets
from datasets import load_dataset, concatenate_datasets

import re


class ConversationDataset(ABC, Dataset):
    """
    Dataset for sentiment analysis.
    """

    def __init__(
            self,
            dataset_id: str,
            tokenizer: transformers.PreTrainedTokenizer
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


class OpenAssistantGuanacoDatasetDict:
    def __init__(
            self,
            tokenizer,
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

        self.train = OpenAssistantGuanacoDataset(second_split_raw_dataset["train"], tokenizer, max_len)
        self.validation = OpenAssistantGuanacoDataset(second_split_raw_dataset["test"], tokenizer, max_len)
        self.test = OpenAssistantGuanacoDataset(first_split_raw_dataset["test"], tokenizer, max_len)


class OpenAssistantGuanacoDataset(ConversationDataset):
    """
    Dataset class to use OpenAssistant-Guanaco dataset with Pytorch and Pytorch Lightning.

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
        dataset (list):
            Tokenized dataset.
        sep_regex (re.Pattern):
            Regular expression to separate the different turns of the dialogues.
    """

    def __init__(
            self,
            raw_dataset: datasets.Dataset,
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 512,
            **kwargs
    ):
        super().__init__(
            "timdettmers/openassistant-guanaco",
            tokenizer
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = []

        self.sep_regex = re.compile(r"### (Human|Assistant)")
        self.preprocess(raw_dataset)

    def preprocess(
            self,
            raw_dataset,
            **kwargs
    ) -> None:
        """
        Preprocesses the dataset.
        It splits the text in sequential turns and inserts the utterances and the role of the speaker in a list of
        dictionaries. Then the chat template of the tokenizer is applied to the list of dictionaries in order to obtain
        a text formatted in the proper way.
        Finally, the text is tokenized and padded to the maximum length.

        Args:
            raw_dataset (datasets.Dataset):
                Raw dataset.
            kwargs:
                Additional keyword arguments.
        """

        texts = [
            self.tokenizer.decode(
                self.tokenizer.apply_chat_template([
                    {
                        "role": "user" if message.split(":", 1)[0].strip() == "Human" else "assistant",
                        "content": message.split(":", 1)[1].strip()
                    } for message in self.sep_regex.sub("<sep/> \\1", dialogue["text"]).split("<sep/>") if
                    len(message) > 0
                ]),
                skip_special_tokens=False).strip() + self.tokenizer.eos_token
            for dialogue in raw_dataset
        ]

        tokenized_texts = [self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True
        ) for text in texts]

        self.dataset = [
            {
                "input_ids": input_encoding["input_ids"].squeeze(0),
                "labels": input_encoding["input_ids"].clone().squeeze(0),
                "attention_mask": input_encoding["attention_mask"].squeeze(0)
            }
            for input_encoding in tokenized_texts]

        for idx, _ in enumerate(self.dataset):
            self.dataset[idx]["labels"][["attention_mask"] == 0] = -100

    def __len__(
            self
    ):
        """
        Returns the length of the dataset.

        Returns:
            int:
                Length of the dataset.
        """

        return len(self.dataset)

    def __getitem__(
            self,
            idx
    ) -> dict:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            dict:
                Dictionary containing the tokenized inputs.
        """

        return self.dataset[idx]


class OpenAssistantGuanacoDataModule(pl.LightningDataModule):
    """
    DataModule for the OpenAssistant-Guanaco dataset.

    Args:
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
        train (OpenAssistantGuanacoDataset):
            Training dataset.
        validation (OpenAssistantGuanacoDataset):
            Validation dataset.
        test (OpenAssistantGuanacoDataset):
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
            "timdettmers/openassistant-guanaco",
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
            "timdettmers/openassistant-guanaco",
        )

        concatenated_raw_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])

        first_split_raw_dataset = concatenated_raw_dataset.train_test_split(
            test_size=self.split[2]
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=self.split[1]/((1-self.split[2]) if self.split[2] != 1 else 1)
        )

        self.train = OpenAssistantGuanacoDataset(
            second_split_raw_dataset["train"],
            self.tokenizer,
            self.max_len
        )
        self.validation = OpenAssistantGuanacoDataset(
            second_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )
        self.test = OpenAssistantGuanacoDataset(
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


if __name__ == "__main__":

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    dataset = OpenAssistantGuanacoDataModule(
        32,
        2,
        tokenizer,
        512,
        (0.8, 0.1, 0.1),
        seed=42
    )
    dataset.setup()

    print(dataset.train)
    print()
