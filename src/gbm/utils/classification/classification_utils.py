import transformers
from transformers import AutoModelForSequenceClassification

from gbm.utils.pipeline.config import Config


def load_original_model_for_sequence_classification(
        config: Config
) -> transformers.AutoModel:
    """
    Loads the model that will be used to experiment alternative architectures.

    Args:
        config (dict):
            The configuration parameters to use in the loading.

    Returns:
        transformers.AutoModel:
            The loaded model.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        config.get("original_model_id"),
        num_labels=config.get("num_classes"),
        id2label=config.get("id2label"),
        label2id=config.get("label2id"),
    )
