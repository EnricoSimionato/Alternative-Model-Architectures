import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from neuroflex.utils.experiment_pipeline.config import Config
from neuroflex.utils.printing_utils.printing_utils import Verbose


def load_original_model_for_sequence_classification(
        config: Config,
) -> transformers.AutoModel:
    """
    Loads the model to be used in the sequence classification task.

    Args:
        config (dict):
            The configuration parameters to use in the loading.

    Returns:
        transformers.AutoModel:
            The model for sequence classification.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        config.get("original_model_id"),
        num_labels=config.get("num_classes"),
        id2label=config.get("id2label"),
        label2id=config.get("label2id"),
    )


def load_tokenizer_for_sequence_classification(
        config: Config,
) -> transformers.AutoModel:
    """
    Loads the tokenizer to be used in the sequence classification task.

    Args:
        config (dict):
            The configuration parameters to use in the loading.

    Returns:
        transformers.AutoTokenizer:
            The tokenizer for sequence classification.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_id")
    )

    if "bert" in config.get("original_model_id"):
        tokenizer.bos_token = "[CLS]"
        tokenizer.eos_token = "[SEP]"

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer