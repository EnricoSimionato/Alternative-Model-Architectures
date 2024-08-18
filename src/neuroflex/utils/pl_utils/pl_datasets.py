from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, PreTrainedTokenizer


class NLPDataset(ABC, Dataset):
    """
    Abstract class to represent a dataset for NLP tasks.

    Args:
        dataset_id (str):
            The dataset id.
        tokenizer (AutoTokenizer | PreTrainedTokenizer):
            The tokenizer to use.

    Attributes:
        dataset_id (str):
            The dataset id.
        tokenizer (AutoTokenizer | PreTrainedTokenizer):
            The tokenizer to use.
    """

    def __init__(
            self,
            dataset_id: str,
            tokenizer: AutoTokenizer | PreTrainedTokenizer
    ) -> None:
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

