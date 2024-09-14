"""
New
"""
import os
import sys
import logging
import pickle as pkl

import torch

from neuroflex.utils.experiment_pipeline.storage_utils import check_path_to_storage
from neuroflex.utils.lm_eval_pipeline.lm_eval_pipeline import evaluate_model_on_benchmark

"""
Old
"""

import argparse

from neuroflex.utils.device_utils import get_available_device
from neuroflex.utils.experiment_pipeline.config import Config, get_path_to_configurations, environments
from neuroflex.utils.experiment_pipeline.experiment import Experiment

from neuroflex.utils.classification import (
    load_model_for_sequence_classification,
    load_tokenizer_for_sequence_classification,
    IMDBDataModule
)
from neuroflex.utils.chatbot import (
    load_model_for_causal_lm,
    load_tokenizer_for_causal_lm,
    OpenAssistantGuanacoDataModule
)

from neuroflex.utils.adapters_utils.adapters_utils import get_adapted_model

from neuroflex.utils.factorized_models_utils.factorized_models_utils import get_factorized_model

from neuroflex.models.factorized_model import KFCTrainedModel, KFCAlphaTrainedModel
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

    original_model = load_model_for_sequence_classification(config)
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

    original_model = load_model_for_causal_lm(config)
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

    original_model = load_model_for_sequence_classification(config)
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

    original_model = load_model_for_causal_lm(config)
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

    original_model = load_model_for_sequence_classification(config)
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

    original_model = load_model_for_causal_lm(config)
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


benchmark_id_eval_args_default_mapping = {

}


def launch_benchmark_evaluation(
        config: Config,
        data: dict
) -> None:
    """
    Launches the benchmark evaluation on a model.

    Args:
        config (Config):
            The configuration parameters for the experiment.
        data (dict):
            The data to use for the experiment.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_layer_redundancy_analysis in redundancy_hunter.py.")
    benchmark_ids = config.get("benchmark_ids")

    if data is None:
        # Getting the parameters from the configuration
        device = get_available_device(config.get("device") if config.contains("device") else None, just_string=True)
        evaluation_args = (config.get("evaluation_args")
                           if config.contains("evaluation_args")
                           else {benchmark_id: {} for benchmark_id in benchmark_ids})

        performance_dict = {benchmark_id: {} for benchmark_id in benchmark_ids}

        # Loading the model and the tokenizer
        base_model = load_model_for_causal_lm(config)
        logger.info(f"Model loaded.")
        print(base_model)
        prepared_model = get_factorized_model(base_model, config)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = load_tokenizer_for_causal_lm(config)
        logger.info(f"Tokenizer loaded.")

        already_created_performance_dict = data
        performance_dict.update(already_created_performance_dict)
        logger.info(f"Previous data loaded.\nLoaded data: {performance_dict}")
        print("Previous data loaded.")

        for benchmark_id in benchmark_ids:
            logger.info(f"Starting the evaluation for the benchmark: {benchmark_id}.")
            print(f"Starting the evaluation for the benchmark: {benchmark_id}.")

            # Defining the evaluation parameters
            default_evaluation_args = (benchmark_id_eval_args_default_mapping[benchmark_id]
                                       if benchmark_id in benchmark_id_eval_args_default_mapping.keys() else {})
            default_evaluation_args.update(
                evaluation_args[benchmark_id] if benchmark_id in evaluation_args.keys() else {}
            )
            evaluation_args = default_evaluation_args
            logger.info(f"Evaluation args: {evaluation_args}")

            # Evaluating the model
            results = evaluate_model_on_benchmark(prepared_model, tokenizer, benchmark_id, evaluation_args, device)

            performance_dict[benchmark_id] = results
            logger.info(f"Performance dictionary updated with the results.")

            logger.info(f"Trying to store the data for benchmark {benchmark_id}...")
            # Saving the data
            with open(config.get("file_path"), "wb") as f:
                pkl.dump((performance_dict, prepared_model), f)
            logger.info(f"Partial data stored.")

            torch.cuda.empty_cache()

        logger.info(f"Trying to store all the data...")
        # Saving the data
        with open(config.get("file_path"), "wb") as f:
            pkl.dump((performance_dict, prepared_model), f)

    performance_dict, prepared_model = data

    # Printing the results
    print("The configuration of the experiment is the following")
    print(config)
    for benchmark_id in performance_dict:
        print(f"The performance of the model on the benchmark {benchmark_id} is {performance_dict[benchmark_id]}")


experiment_mapping = {
    "benchmark_evaluation": launch_benchmark_evaluation,
}

specific_mandatory_keys_mapping = {

}

not_used_keys_mapping = {

}


def main() -> None:
    """
    Main method to start the various types of analyses on a deep model.
    """

    if len(sys.argv) < 3:
        raise Exception("Please provide the name of the configuration file and the environment.\n"
                        "Example: python rank_analysis_launcher.py config_name environment"
                        "'environment' can be 'local' or 'server' or 'colab'.")

    # Extracting the configuration name and the environment
    config_name = sys.argv[1]
    environment = sys.argv[2]

    # Loading the configuration
    configuration = Config(os.path.join(get_path_to_configurations(environment), "aa", config_name))

    # Checking if the configuration file contains the necessary keys
    mandatory_keys = [
        "path_to_storage",
        "experiment_type",
        "model_id",
    ]
    configuration.check_mandatory_keys(mandatory_keys)
    experiment_type = configuration.get("experiment_type")
    mandatory_keys += (specific_mandatory_keys_mapping[experiment_type]
                       if experiment_type in specific_mandatory_keys_mapping.keys()
                       else [])
    not_used_keys = not_used_keys_mapping[experiment_type] if experiment_type in not_used_keys_mapping.keys() else []
    mandatory_keys = list(set(mandatory_keys) - set(not_used_keys))
    configuration.check_mandatory_keys(mandatory_keys)

    # Checking the path to the storage
    file_available, directory_path, file_name = check_path_to_storage(
        configuration.get("path_to_storage"),
        configuration.get("experiment_type"),
        configuration.get("model_id").split("/")[-1],
        configuration.get("version") if configuration.contains("version") else None,
    )
    configuration.update(
        {
            "file_available": file_available,
            "file_path": os.path.join(directory_path, file_name),
            "directory_path": directory_path,
            "file_name": file_name,
            "log_path": os.path.join(directory_path, "logs.log")
        }
    )

    # Storing the configuration
    configuration.store(configuration.get("directory_path"))

    # Creating the logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(configuration.get("log_path"))]
    )
    logger = logging.getLogger()
    logger.info(f"Running main in experiment_launcher.py.")
    logger.info(f"Environment: {environment}.")
    logger.info(f"Configuration file: {config_name}.")

    # Checking if the analysis type is recognized
    if experiment_type not in experiment_mapping.keys():
        logger.error(f"The experiment type is not recognized.")
        raise Exception("The experiment type is not recognized.")

    # Loading the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{configuration.get('file_path')}' is available.")
        with open(configuration.get('file_path'), "rb") as f:
            data = pkl.load(f)
    else:
        data = None

    # Performing the analysis
    logger.info(f"Starting the experiment {experiment_type}.")
    experiment_mapping[experiment_type](configuration, data)

    # Storing the configuration
    configuration.store(configuration.get("directory_path"))


if __name__ == "__main__":
    main()
