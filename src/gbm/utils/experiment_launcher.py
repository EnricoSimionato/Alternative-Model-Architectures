import os

from gbm.utils import (
    Config,
    Experiment
)

from gbm.utils.adapters_utils.adapters_utils import get_adapted_model

from gbm.utils.classification import (
    load_original_model_for_sequence_classification,
    load_tokenizer_for_sequence_classification,
    IMDBDataModule
)
from gbm.utils.chatbot import (
    load_original_model_for_causal_lm,
    load_tokenizer_for_causal_lm,
    OpenAssistantGuanacoDataModule
)

from gbm.utils.factorized_models_utils.factorized_models_utils import get_factorized_model


def launch_aa_class_experiment(
        config: Config
) -> None:
    """
    Launches an experiment using alternative architectures on a classification task.

    Args:
        config (Config):
            The configuration parameters for the experiment.
    """

    original_model = load_original_model_for_sequence_classification(config)
    tokenizer = load_tokenizer_for_sequence_classification(config)

    if not config.contains("factorization_method"):
        raise ValueError("Factorization method not specified")

    model = get_factorized_model(
        original_model,
        config
    )

    if config.contains("adapter_method"):
        model = get_adapted_model(model, config)
    else:
        model = model

    experiment = Experiment(
        task="classification",
        model=model,
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


def launch_aa_chat_experiment(
        config: Config
) -> None:
    """
    Launches an experiment using alternative architectures on a chatbot task.

    Args:
        config (Config):
            The configuration parameters for the experiment.
    """

    original_model = load_original_model_for_causal_lm(config)
    tokenizer = load_tokenizer_for_causal_lm(config)

    if not config.contains("factorization_method"):
        raise ValueError("Factorization method not specified")

    model = get_factorized_model(
        original_model,
        config
    )

    if config.contains("adapter_method"):
        model = get_adapted_model(model, config)
    else:
        model = model

    experiment = Experiment(
        task="chatbot",
        model=model,
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


def launch_kfc_class_experiment(
        config: Config
) -> None:
    """
    Launches the KFC training experiment on a classification task.

    Args:
        config (Config):
            The configuration parameters for the experiment.
    """

    original_model = load_original_model_for_sequence_classification(config)
    tokenizer = load_tokenizer_for_sequence_classification(config)

    if not config.contains("kfc_training") or not config.get("kfc_training"):
        raise ValueError("KFC training not enabled")
    if not config.contains("adapter_method"):
        raise ValueError("Adapter method needs to be specified to allow KFC training")

    model = get_adapted_model(original_model, config)
    print(model)
    experiment = Experiment(
        task="classification",
        model=model,
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


def launch_kfc_chat_experiment(
        config: Config
) -> None:
    """
    Launches the KFC training experiment on a chatbot task.

    Args:
        config (Config):
            The configuration parameters for the experiment.
    """

    original_model = load_original_model_for_causal_lm(config)
    tokenizer = load_tokenizer_for_causal_lm(config)

    if not config.contains("kfc_training") or not config.get("kfc_training"):
        raise ValueError("KFC training not enabled")
    if not config.contains("adapter_method"):
        raise ValueError("Adapter method needs to be specified to allow KFC training")

    model = get_adapted_model(original_model, config)

    experiment = Experiment(
        task="chatbot",
        model=model,
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
        launch_kfc_chat_experiment(configuration)
    else:
        raise ValueError("Invalid experiment")
