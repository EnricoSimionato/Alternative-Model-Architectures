import os
import json
import sys


def load_experiment_config(path_to_experiment):
    config_path = os.path.join(path_to_experiment, "configuration", "config.json")
    if not os.path.exists(config_path):
        raise Exception(f"Configuration file not found at: '{config_path}'")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 config_printer.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    path_to_experiment = os.path.join(os.getcwd().split("src")[0], "src", "experiments", "performed_experiments", experiment_name)

    if not os.path.exists(path_to_experiment):
        print(f"Experiment '{experiment_name}' not found.")
        sys.exit(1)

    config = load_experiment_config(path_to_experiment)

    print(json.dumps(config, indent=4))


if __name__ == "__main__":
    main()
