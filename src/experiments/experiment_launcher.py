import os

from transformers import AutoTokenizer

from gbm.utils import Config, Experiment

from gbm.utils.classification import load_original_model_for_sequence_classification, IMDBDataModule
from gbm.utils.chatbot import load_original_model_for_causal_lm, OpenAssistantGuanacoDataModule


def launch_aa_class_experiment(config: Config):
    original_model = load_original_model_for_sequence_classification(config)
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_id"))
    if "bert" in config.get("original_model_id"):
        tokenizer.bos_token = "[CLS]"
        tokenizer.eos_token = "[SEP]"

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    experiment = Experiment(
        task="classification",
        model=original_model,
        dataset=IMDBDataModule(
            config.get("batch_size"),
            config.get("num_workers"),
            tokenizer,
            config.get("max_len_tokenizer"),
            config.get("split"),
            seed=config.get("seed")
        ),
        config=config,
        tokenizer=tokenizer
    )

    experiment.run_experiment()


def launch_aa_chat_experiment(config: Config):
    original_model = load_original_model_for_causal_lm(config)
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_id"))
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        config.get("tokenizer_id_for_chat_template")).chat_template

    experiment = Experiment(
        task="chatbot",
        model=original_model,
        dataset=OpenAssistantGuanacoDataModule(
            config.get("batch_size"),
            config.get("num_workers"),
            tokenizer,
            config.get("max_len_tokenizer"),
            config.get("split"),
            seed=config.get("seed")
        ),
        config=config,
        tokenizer=tokenizer
    )

    experiment.run_experiment()


def launch_kfc_class_experiment(config: Config):
    original_model = load_original_model_for_sequence_classification(config)
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_id"))
    if "bert" in config.get("original_model_id"):
        tokenizer.bos_token = "[CLS]"
        tokenizer.eos_token = "[SEP]"

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    experiment = Experiment(
        task="classification",
        model=original_model,
        dataset=IMDBDataModule(
            config.get("batch_size"),
            config.get("num_workers"),
            tokenizer,
            config.get("max_len_tokenizer"),
            config.get("split"),
            seed=config.get("seed")
        ),
        config=config,
        tokenizer=tokenizer
    )

    experiment.run_experiment()


def launch_kfc_chat_experiment(config: Config):
    original_model = load_original_model_for_causal_lm(config)
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_id"))
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        config.get("tokenizer_id_for_chat_template")).chat_template

    experiment = Experiment(
        task="chatbot",
        model=original_model,
        dataset=OpenAssistantGuanacoDataModule(
            config.get("batch_size"),
            config.get("num_workers"),
            tokenizer,
            config.get("max_len_tokenizer"),
            config.get("split"),
            seed=config.get("seed")
        ),
        config=config,
        tokenizer=tokenizer
    )

    experiment.run_experiment()



if __name__ == "__main__":
    config_file_name = "CONFIG_KFC_BERT_CLASS.json"
    #path_to_config = os.path.join("/home/enricosimionato/thesis", config_file_name)
    path_to_config = os.path.join("/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/local", config_file_name)
    configuration = Config(path_to_config)

    if "AA" in config_file_name and "CLASS" in config_file_name:
        launch_aa_class_experiment(configuration)
    elif "KFC" in config_file_name and "CLASS" in config_file_name:
        launch_kfc_class_experiment(configuration)
    elif "AA" in config_file_name and "CHAT" in config_file_name:
        launch_aa_chat_experiment(configuration)
    elif "KFC" in config_file_name and "CHAT" in config_file_name:
        launch_aa_chat_experiment(configuration)
    else:
        raise ValueError("Invalid experiment")
