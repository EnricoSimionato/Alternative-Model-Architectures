import os
import sys

from neuroflex.utils.experiment_pipeline import Config
from neuroflex.utils.experiment_pipeline.config import get_path_to_configurations
from neuroflex.utils.printing_utils.printing_utils import Verbose

from neuroflex.imaage.imaage_training import (
    imaage_train,
    perform_simple_initialization_analysis,
    perform_global_matrices_initialization_analysis
)


def main():
    """
    Main method to start the layers rank analysis
    """

    if len(sys.argv) < 3:
        raise Exception("Please provide the name of the configuration file and the environment.\n"
                        "Example: python rank_analysis_launcher.py config_name environment"
                        "'environment' can be 'local' or 'server' or 'colab'.")

    # Extracting the configuration name and the environment
    config_name = sys.argv[1]
    environment = sys.argv[2]

    # Loading the configuration
    configuration = Config(
        os.path.join(get_path_to_configurations(environment), "imaage", config_name)
    )
    verbose = Verbose(configuration.get("verbose") if configuration.contains("verbose") else 0)

    # Checking if the configuration file contains the necessary keys
    mandatory_keys = [
        "path_to_storage",
        "analysis_type",
        "targets",
        "original_model_id",
        "rank"
    ]
    configuration.check_mandatory_keys(mandatory_keys)

    if configuration.get("analysis_type") == "simple_initialization_analysis":
        # Perform the simple initialization analysis
        perform_simple_initialization_analysis(configuration, verbose)
    elif configuration.get("analysis_type") == "global_matrices_initialization_analysis":
        # Perform the global matrix initialization analysis
        perform_global_matrices_initialization_analysis(configuration, verbose)
    else:
        raise Exception("The analysis type is not recognized.")


if __name__ == "__main__":
    main()
