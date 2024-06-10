from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer

from peft import get_peft_model, LoraConfig

from gbm import GlobalBaseModel
from gbm.utils.classification import load_original_model_for_sequence_classification, IMDBDataModule
from gbm.utils.pipeline.experiment import Experiment
from gbm.utils.pipeline.config import Config
from gbm.utils.chatbot.conversation_utils import load_original_model_for_causal_lm
from gbm.utils.chatbot import OpenAssistantGuanacoDataModule
from gbm.utils.chatbot.conversation_utils import start_conversation_loop

configuration = Config(
    "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_BERT_CLASS.json"
)
original_model = load_original_model_for_causal_lm(configuration)
tokenizer = AutoTokenizer.from_pretrained(configuration.get("tokenizer_id"))

def test_experiment_with_chatbot():
    configuration = Config(
        "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_BERT_CHAT.json"
    )
    original_model = load_original_model_for_causal_lm(configuration)
    tokenizer = AutoTokenizer.from_pretrained(configuration.get("tokenizer_id"))
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        configuration.get("tokenizer_id_for_chat_template")).chat_template

    """
    # Create the global model
    global_model = GlobalBaseModel(
        original_model,
        target_layers={
            "word_embeddings": {"rank": 128},
            "query": {"rank": 64},
            "key": {"rank": 64},
            "value": {"rank": 64},
            "dense": {"rank": 64},
        },
        use_names_as_keys=True,
        remove_average=True,
        preserve_original_model=True,
        verbose=True
    )
    """

    experiment = Experiment(
        task="chatbot",
        model=original_model,
        dataset=OpenAssistantGuanacoDataModule(
            configuration.get("batch_size"),
            configuration.get("num_workers"),
            tokenizer,
            configuration.get("max_len_tokenizer"),
            configuration.get("split")
        ),
        config=configuration,
        tokenizer=tokenizer
    )

    experiment.run_experiment()

def test_experiment_with_classifier():
    configuration = Config(
        "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_BERT_CLASS.json"
    )
    original_model = load_original_model_for_causal_lm(configuration)
    tokenizer = AutoTokenizer.from_pretrained(configuration.get("tokenizer_id"))
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        configuration.get("tokenizer_id_for_chat_template")).chat_template

    # Create the global model
    global_model = GlobalBaseModel(
        original_model,
        target_layers={
            "word_embeddings": {"rank": 128},
            "query": {"rank": 64},
            "key": {"rank": 64},
            "value": {"rank": 64},
            "dense": {"rank": 64},
        },
        use_names_as_keys=True,
        remove_average=True,
        preserve_original_model=True,
        verbose=True
    )

    experiment = Experiment(
        task="chatbot",
        model=global_model,
        dataset=OpenAssistantGuanacoDataModule(
            configuration.get("batch_size"),
            configuration.get("num_workers"),
            tokenizer,
            configuration.get("max_len_tokenizer"),
            configuration.get("split")
        ),
        config=configuration,
        tokenizer=tokenizer
    )

    experiment.run_experiment()


def test_storage_utilities_of_experiment():
    experiment = Experiment(
        task="chatbot",
        model=original_model,
        dataset=OpenAssistantGuanacoDataModule(
            configuration.get("batch_size"),
            configuration.get("num_workers"),
            tokenizer,
            configuration.get("max_len_tokenizer"),
            configuration.get("split")
        ),
        config=configuration,
        tokenizer=tokenizer
    )
    experiment.start_experiment()

    experiment.store_experiment()

    Experiment.load_experiment(configuration.get("path_to_experiment"))


    start_conversation_loop(
        experiment.get_model(),
        experiment.get_tokenizer(),
        experiment.get_config().get("stop_tokens"),
        user_inputs=["Hello", "How are you?"]
    )


def test_kfc():
    configuration = Config(
        "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_BERT_KFC_CLASS.json"
    )
    original_model = load_original_model_for_sequence_classification(configuration)
    tokenizer = AutoTokenizer.from_pretrained(configuration.get("tokenizer_id"))
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    lora_configuration = LoraConfig(
        r=16,
        target_modules="all-linear"
    )

    model = get_peft_model(
        original_model,
        lora_configuration
    )

    for name, param in model.named_parameters():
        param.requires_grad = True

    experiment = Experiment(
        task="classification",
        model=model,
        dataset=IMDBDataModule(
            configuration.get("batch_size"),
            configuration.get("num_workers"),
            tokenizer,
            configuration.get("max_len_tokenizer"),
            configuration.get("split"),
            configuration.get("seed")
        ),
        config=configuration,
        tokenizer=tokenizer
    )

    experiment.run_experiment()


if __name__ == "__main__":

    test_kfc()
