from __future__ import annotations

from abc import ABC, abstractmethod

import datasets
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from datasets import load_dataset, concatenate_datasets

import re


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


class OpenAssistantGuanacoDatasetDict:
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

        self.train = OpenAssistantGuanacoDataset(second_split_raw_dataset["train"], tokenizer, max_len)
        self.validation = OpenAssistantGuanacoDataset(second_split_raw_dataset["test"], tokenizer, max_len)
        self.test = OpenAssistantGuanacoDataset(first_split_raw_dataset["test"], tokenizer, max_len)


class OpenAssistantGuanacoDataset(ConversationDataset):
    def __init__(
            self,
            raw_dataset,
            tokenizer,
            max_length: int = 512
    ):
        super().__init__("openassistant-guanaco", tokenizer)

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = []

        self.sep_regex = re.compile(r"### (Human|Assistant)")
        self.preprocess(raw_dataset)

    def preprocess(
            self,
            raw_dataset
    ) -> None:
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


if __name__ == "__main__":
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = IMDBDatasetDict(
        bert_tokenizer,
        512,
        (0.8, 0.1, 0.1)
    )

    dl = DataLoader(dataset.validation, batch_size=16, shuffle=True)

    for batch in dl:
        print(batch["label"])

