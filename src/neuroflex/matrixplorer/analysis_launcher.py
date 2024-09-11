import os
import sys
import logging

from neuroflex.matrixplorer.activations_analysis import perform_activations_analysis, perform_delta_activations_analysis
from neuroflex.matrixplorer.head_analysis import perform_head_analysis, perform_heads_similarity_analysis
from neuroflex.matrixplorer.sorted_layers_rank_analysis import perform_sorted_layers_rank_analysis
from neuroflex.redhunter.rendundancy_hunter import perform_layer_redundancy_analysis_launcher
from neuroflex.utils.experiment_pipeline import Config
from neuroflex.utils.experiment_pipeline.config import get_path_to_configurations
from neuroflex.utils.experiment_pipeline.storage_utils import check_path_to_storage

from neuroflex.matrixplorer.matrix_initialization_analysis import (
    perform_simple_initialization_analysis,
    perform_global_matrices_initialization_analysis
)


analysis_mapping = {
    "simple_initialization_analysis": perform_simple_initialization_analysis,
    "global_matrices_initialization_analysis": perform_global_matrices_initialization_analysis,

    "sorted_layers_rank_analysis": perform_sorted_layers_rank_analysis,

    "head_analysis": perform_head_analysis,
    "heads_similarity_analysis": perform_heads_similarity_analysis,

    "activations_analysis": perform_activations_analysis,
    "delta_activations_analysis": perform_delta_activations_analysis,

    "layer_redundancy_analysis": perform_layer_redundancy_analysis_launcher
}

specific_mandatory_keys_mapping = {
    "simple_initialization_analysis": ["rank"],
    "global_matrices_initialization_analysis": ["rank"],

    "sorted_layers_rank_analysis": ["explained_variance_threshold"],

    "head_analysis": ["explained_variance_threshold", "name_num_heads_mapping"],
    "heads_similarity_analysis": ["grouping"],

    "activations_analysis": ["dataset_id"],
    "delta_activations_analysis": ["dataset_id"],

    "query_key_analysis": ["query_label", "key_label"],
    "layer_redundancy_analysis": ["num_layers"]
}

not_used_keys_mapping = {
    "query_key_analysis": ["targets", "black_list"],
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
    configuration = Config(os.path.join(get_path_to_configurations(environment), "analysis", config_name))

    # Checking if the configuration file contains the necessary keys
    mandatory_keys = [
        "path_to_storage",
        "analysis_type",
        "targets",
        "original_model_id",
    ]
    configuration.check_mandatory_keys(mandatory_keys)
    mandatory_keys += specific_mandatory_keys_mapping[configuration.get("analysis_type")] if configuration.get("analysis_type") in specific_mandatory_keys_mapping.keys() else []
    mandatory_keys = list(set(mandatory_keys) - set(not_used_keys_mapping[configuration.get("analysis_type")] if configuration.get("analysis_type") in not_used_keys_mapping.keys() else []))
    configuration.check_mandatory_keys(mandatory_keys)

    # Checking the path to the storage
    file_available, directory_path, file_name = check_path_to_storage(
        configuration.get("path_to_storage"),
        configuration.get("analysis_type"),
        configuration.get("original_model_id").split("/")[-1],
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
    print(configuration.get("log_path"))
    # Storing the configuration
    configuration.store(configuration.get("directory_path"))

    # Creating the logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(configuration.get("log_path"))
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Running main in analysis_launcher.py.")
    logger.info(f"Configuration file: {config_name}.")
    logger.info(f"Environment: {environment}.")

    # Checking if the analysis type is recognized
    if configuration.get("analysis_type") not in analysis_mapping.keys():
        logger.error(f"The analysis type is not recognized.")
        raise Exception("The analysis type is not recognized.")

    # Performing the analysis
    logger.info(f"Starting the analysis {configuration.get('analysis_type')}.")
    analysis_mapping[configuration.get("analysis_type")](configuration)

    # Storing the configuration
    configuration.store(configuration.get("directory_path"))


if __name__ == "__main__":
    main()
