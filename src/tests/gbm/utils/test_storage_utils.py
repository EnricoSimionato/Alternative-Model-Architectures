from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from gbm.utils.storage_utils import store_model_and_info, load_model_and_info
from gbm.models.global_dependent_model import GlobalBaseModel

CONFIG = {
    "original_model_id": "bert-base-cased",
    "quantization": None,

    "num_classes": 2,
    "id2label": {0: "NEGATIVE", 1: "POSITIVE"},
    "label2id": {"NEGATIVE": 0, "POSITIVE": 1},

    "dataset_id": "imdb",
    "max_len_samples": 16384,
    "split": (0.8, 0.1, 0.1),
    "tokenizer_id": "bert-base-cased",
    "max_len_tokenizer": 386,

    "num_epochs": 5,
    "learning_rate": 1e-5,
    "lr_schedule": "linear",
    "batch_size": 16,
    "num_workers": 2,
    "warmup": 0.2,
    "num_checks_per_epoch": 2,
    "gradient_accumulation_steps": 4,

    "target_layers": {
        "query": {"rank": 128},
        "key": {"rank": 64},
        "value": {"rank": 256},
        "dense": {"rank": 64}
        },

    "original_model_parameters": 0,
    "model_parameters": 0,
    "percentage_parameters": 0,
    "model_trainable_parameters": 0,
}
MODEL = GlobalBaseModel(
    AutoModelForSequenceClassification.from_pretrained(CONFIG["original_model_id"]),
    target_layers=CONFIG["target_layers"],
)
TOKENIZER = AutoTokenizer.from_pretrained(CONFIG["tokenizer_id"])
LOSSES = {}
PATH_TO_STORAGE = "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/tests/test_data"


def test_store_model_and_info():
    model_name = store_model_and_info(
        f"GlobalBaseModel-{CONFIG['original_model_id']}",
        MODEL,
        TOKENIZER,
        CONFIG,
        LOSSES,
        PATH_TO_STORAGE,
        verbose=True
    )

    print(f"Model stored at: {model_name}")

def test_load_model_and_info():
    model_name = "GlobalBaseModel-bert-base-cased_2024_05_07_15_54_21"

    model, tokenizer, config, losses = load_model_and_info(
        model_name,
        PATH_TO_STORAGE,
        verbose=True
    )

    print(f"Model loaded from: {model}")
    print(f"Tokenizer loaded from: {tokenizer}")
    print(f"Config loaded from: {config}")
    print(f"Losses loaded from: {losses}")