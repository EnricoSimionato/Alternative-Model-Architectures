import os

from gbm.utils.experiment_pipeline.config import Config
from gbm.utils.experiment_pipeline.experiment import Experiment

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

from gbm.utils.adapters_utils.adapters_utils import get_adapted_model
from gbm.utils.factorized_models_utils.factorized_models_utils import get_factorized_model

from gbm.models.global_dependent_model import KFCTrainedModel


keys_for_regularized_training = [
    "initial_regularization_weight",
    "max_regularization_weight",
    "start_step_regularization",
    "steps_regularization_weight_resets"
]


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

    factorized_model = get_factorized_model(
        original_model,
        config
    )
    print(factorized_model.__dict__)
    print(factorized_model.model)

    experiment = Experiment(
        task="classification",
        model=factorized_model,
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

    factorized_model = get_factorized_model(
        original_model,
        config
    )

    experiment = Experiment(
        task="chatbot",
        model=factorized_model,
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

    if not config.contains("adapter_method"):
        raise ValueError("Adapter method needs to be specified to allow KFC training")

    model = get_adapted_model(original_model, config)
    print(model)
    regularized_training_arguments = config.get_dict(keys_for_regularized_training)
    kfc_wrapped_model = KFCTrainedModel(
        model,
        **regularized_training_arguments
    )
    print(model)

    experiment = Experiment(
        task="classification",
        model=kfc_wrapped_model,
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

    if not config.contains("adapter_method"):
        raise ValueError("Adapter method needs to be specified to allow KFC training")

    model = get_adapted_model(original_model, config)
    print(model)
    regularized_training_arguments = config.get_dict(keys_for_regularized_training)
    kfc_wrapped_model = KFCTrainedModel(
        model,
        **regularized_training_arguments
    )
    print(model)

    experiment = Experiment(
        task="chatbot",
        model=kfc_wrapped_model,
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
    config_file_name = "CONFIG_AA_BERT_CLASS.json"
    #path_to_config = os.path.join("/home/enricosimionato/thesis", config_file_name)
    path_to_config = os.path.join("/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/local", config_file_name)
    configuration = Config(path_to_config)
    print(f"Running experiment with configuration: {config_file_name}")

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
