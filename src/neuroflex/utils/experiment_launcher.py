import os
import argparse

from neuroflex.utils.experiment_pipeline.config import Config, get_path_to_configurations, environments
from neuroflex.utils.experiment_pipeline.experiment import Experiment

from neuroflex.utils.classification import (
    load_original_model_for_sequence_classification,
    load_tokenizer_for_sequence_classification,
    IMDBDataModule
)
from neuroflex.utils.chatbot import (
    load_original_model_for_causal_lm,
    load_tokenizer_for_causal_lm,
    OpenAssistantGuanacoDataModule
)

from neuroflex.utils.adapters_utils.adapters_utils import get_adapted_model
from neuroflex.utils.factorized_models_utils.factorized_models_utils import get_factorized_model

from neuroflex.models.global_dependent_model import KFCTrainedModel, KFCAlphaTrainedModel
from neuroflex.utils.plotting_utils.heatmap import create_heatmap_global_layers

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
    print(original_model)

    factorized_model = get_factorized_model(
        original_model,
        config
    )
    print(factorized_model)

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
    print(original_model)

    factorized_model = get_factorized_model(
        original_model,
        config
    )
    print(factorized_model)

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
    print(original_model)

    if not config.contains("adapter_method"):
        raise ValueError("Adapter method needs to be specified to allow KFC training")

    model = get_adapted_model(original_model, config)
    regularized_training_arguments = config.get_dict(keys_for_regularized_training)
    kfc_wrapped_model = KFCTrainedModel(
        model,
        **regularized_training_arguments
    )
    print(kfc_wrapped_model)

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
    print

    if not config.contains("adapter_method"):
        raise ValueError("Adapter method needs to be specified to allow KFC training")

    model = get_adapted_model(original_model, config)
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


def launch_kfc_alpha_class_experiment(
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

    model = get_adapted_model(original_model, config)
    kfc_alpha_training_arguments = config.get_dict(["initial_alpha", "horizon"])
    kfc_alpha_wrapped_model = KFCAlphaTrainedModel(
        model,
        **kfc_alpha_training_arguments
    )
    print(kfc_alpha_wrapped_model)

    experiment = Experiment(
        task="classification",
        model=kfc_alpha_wrapped_model,
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


def launch_kfc_alpha_chat_experiment(
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


def main():
    """
    Main function to run the experiment with the specified configuration.
    """

    parser = argparse.ArgumentParser(description='Run experiment with specified configuration.')
    parser.add_argument(
        "config_file_name",
        type=str,
        help="The name of the configuration file to use for the experiment (e.g., 'CONFIG.json')."
    )
    parser.add_argument(
        "environment",
        type=str,
        choices=environments,
        help=f"Specify the environment: {' or '.join(environments)}."
    )

    # Extracting the configuration name and the environment
    args = parser.parse_args()
    config_file_name = args.config_file_name
    environment = args.environment
    print(f"Running experiment with configuration: {config_file_name} on {environment} environment")

    # Loading the configuration
    configuration = Config(os.path.join(get_path_to_configurations(environment), "aa", config_file_name))

    if "CLASS" in config_file_name:
        if "AA" in config_file_name:
            launch_aa_class_experiment(configuration)
        elif "KFC" in config_file_name and "ALPHA" in config_file_name:
            launch_kfc_alpha_class_experiment(configuration)
        elif "KFC" in config_file_name:
            launch_kfc_class_experiment(configuration)
        else:
            raise ValueError("Invalid experiment")
    elif "CHAT" in config_file_name:
        if "AA" in config_file_name:
            launch_aa_chat_experiment(configuration)
        elif "KFC" in config_file_name and "ALPHA" in config_file_name:
            launch_kfc_alpha_chat_experiment(configuration)
        elif "KFC" in config_file_name:
            launch_kfc_chat_experiment(configuration)
        else:
            raise ValueError("Invalid experiment")
    else:
        raise ValueError("Invalid experiment")


if __name__ == "__main__":
    main()
