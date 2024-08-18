import os
import sys

from gbm.utils.experiment_pipeline import Config
from gbm.utils.experiment_pipeline.config import get_path_to_configurations
from gbm.utils.printing_utils.printing_utils import Verbose

from gbm.imaage.imaage_training import imaage_train, imaage_training_trial


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
        "targets",
        "original_model_id",
        "rank"
    ]
    configuration.check_mandatory_keys(mandatory_keys)

    # Performing the rank analysis
    imaage_training_trial(configuration, verbose)


if __name__ == "__main__":
    main()
