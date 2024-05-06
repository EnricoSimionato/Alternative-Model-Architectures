import transformers
from transformers import AutoModelForSequenceClassification


def load_original_model_for_sequence_classification(
        config: dict
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
        config["original_model_id"],
        num_labels=config["num_classes"],
        id2label=config["id2label"],
        label2id=config["label2id"],
    )
