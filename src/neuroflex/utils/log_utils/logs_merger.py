import argparse
import os
import shutil


def merge_logs(
    model_filters: tuple = (),
    experiment_filters: tuple = ()
) -> None:
    """
    Merges TensorBoard logs from different experiments into a single directory.

    Args:
        model_filters (tuple, optional):
            A tuple of model names to filter the logs.
            Paths containing any of the model names will be copied.
            If empty, no filtering will be applied.
        experiment_filters (tuple, optional):
            A tuple of experiment names to filter the logs.
            Paths containing any of the experiment names will be copied.
            If empty, no filtering will be applied.
    """

    # Directories are defined starting inside the root directory of the project (Alternative-Model-Architectures)
    results_directory = os.path.join(os.getcwd(), "src", "experiments", "results")
    destination_directory = os.path.join(results_directory, 'all_logs')

    # Ensuring the destination directory exists and is empty.
    # If it doesn't exist, create it.
    # If it does, delete it and create it again.
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    else:
        shutil.rmtree(destination_directory)
        os.makedirs(destination_directory)

    # Walking through the directory structure
    for root, dirs, files in os.walk(results_directory):
        for dir_name in dirs:
            if "tensorboard_logs" in dir_name:
                log_path = os.path.join(root, dir_name)

                # Applying filters if they are specified
                if model_filters or experiment_filters:
                    # Skipping if filters don't match
                    if model_filters and not any(model in log_path for model in model_filters):
                        continue
                    if experiment_filters and not any(exp in log_path for exp in experiment_filters):
                        continue

                # Handling `version_x` directories inside `tensorboard_logs`
                for version_dir in os.listdir(log_path):
                    version_path = os.path.join(log_path, version_dir)
                    if os.path.isdir(version_path):
                        relative_path = os.path.relpath(version_path, results_directory)
                        dest_path = os.path.join(destination_directory, relative_path)

                        # Creating destination directory structure
                        os.makedirs(dest_path, exist_ok=True)

                        # Copying files from the `version_x` directory
                        for log_file in os.listdir(version_path):
                            src_file = os.path.join(version_path, log_file)
                            dest_file = os.path.join(dest_path, log_file)
                            shutil.copy(src_file, dest_file)

                        print(f"Copied logs from {version_path} to {dest_path}")


if __name__ == "__main__":
    # Setting up argument parsing
    parser = argparse.ArgumentParser(description="Merge TensorBoard logs with optional filters.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=(),
        help="List of model filters (space-separated). Leave empty for no filtering."
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=(),
        help="List of experiment filters (space-separated). Leave empty for no filtering."
    )
    args = parser.parse_args()

    merge_logs(model_filters=args.models, experiment_filters=args.experiments)