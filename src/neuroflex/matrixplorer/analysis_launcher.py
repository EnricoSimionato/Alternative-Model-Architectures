import os
import sys

from neuroflex.matrixplorer.head_analysis import perform_head_analysis, perform_heads_similarity_analysis
from neuroflex.utils.experiment_pipeline import Config
from neuroflex.utils.experiment_pipeline.config import get_path_to_configurations
from neuroflex.utils.experiment_pipeline.storage_utils import check_path_to_storage
from neuroflex.utils.printing_utils.printing_utils import Verbose

from neuroflex.matrixplorer.matrix_initialization_analysis import (
    perform_simple_initialization_analysis,
    perform_global_matrices_initialization_analysis
)


analysis_mapping = {
    "simple_initialization_analysis": perform_simple_initialization_analysis,
    "global_matrices_initialization_analysis": perform_global_matrices_initialization_analysis,

    "head_analysis": perform_head_analysis,
    "heads_similarity_analysis": perform_heads_similarity_analysis
}

specific_mandatory_keys_mapping = {
    "simple_initialization_analysis": ["rank"],
    "global_matrices_initialization_analysis": ["rank"],

    "head_analysis": ["explained_variance_threshold", "name_num_heads_mapping"],
    "heads_similarity_analysis": []
}


def append_additional_words_to_be_in_the_file_name(
        configuration,
        words_to_be_in_the_file_name
) -> list[str]:
    """
    Appends the additional words to be in the file name to the list of words to be in the file name.

    Args:
        configuration (Config):
            The configuration object containing the necessary information to perform the analysis.
        words_to_be_in_the_file_name (list[str]):
            The list of words to be in the file name.

    Returns:
        list[str]:
            The updated list of words to be in the file name.

    """

    analysis_type = configuration.get("analysis_type")

    if analysis_type not in analysis_mapping.keys():
        raise Exception("The analysis type is not recognized.")
    if analysis_type == "simple_initialization_analysis":
        words_to_be_in_the_file_name += ["rank"] + [str(configuration.get("rank"))]
    elif analysis_type == "global_matrices_initialization_analysis":
        words_to_be_in_the_file_name += ["rank"] + [str(configuration.get("rank"))]
    elif analysis_type == "head_analysis":
        words_to_be_in_the_file_name += ["explained_variance_threshold"] + [str(configuration.get("explained_variance_threshold"))]

    return words_to_be_in_the_file_name


def main():
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
    verbose = Verbose(configuration.get("verbose") if configuration.contains("verbose") else 0)

    # Checking if the configuration file contains the necessary keys
    mandatory_keys = [
        "path_to_storage",
        "analysis_type",
        "targets",
        "original_model_id",
    ]
    configuration.check_mandatory_keys(mandatory_keys)
    mandatory_keys += specific_mandatory_keys_mapping[configuration.get("analysis_type")]
    configuration.check_mandatory_keys(mandatory_keys)

    # Setting the words to be in the file name
    words_to_be_in_the_file_name = (
            ["paths"] + configuration.get("targets") +
            ["black_list"] + configuration.get("black_list")
    )
    words_to_be_in_the_file_name = append_additional_words_to_be_in_the_file_name(configuration, words_to_be_in_the_file_name)
    for index in range(len(words_to_be_in_the_file_name)):
        words_to_be_in_the_file_name[index] = words_to_be_in_the_file_name[index].replace(".", "_").replace("/", "_")

    # Checking the path to the storage
    file_available, directory_path, file_name = check_path_to_storage(
        configuration.get("path_to_storage"),
        configuration.get("analysis_type"),
        configuration.get("original_model_id").split("/")[-1],
        tuple(words_to_be_in_the_file_name)
    )
    configuration.update(
        {
            "file_available": file_available,
            "file_path": os.path.join(directory_path, file_name),
            "directory_path": directory_path,
            "file_name": file_name,
            "file_name_no_format": file_name.split(".")[0]
        }
    )

    # Performing the analysis
    if configuration.get("analysis_type") not in analysis_mapping.keys():
        raise Exception("The analysis type is not recognized.")
    analysis_mapping[configuration.get("analysis_type")](configuration)


if __name__ == "__main__":
    main()
