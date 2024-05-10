import os
from datetime import datetime
import json
from typing import Any


class Config:
    """
    Config class to store all the configuration parameters.

    Args:
        path_to_config (str):
            The path to the configuration file.

    Attributes:
        All the keys in the config file are added as attributes to the class.
        begin_time (str):
            The time when the experiment started.
        end_time (str):
            The time when the experiment ended.
    """

    def __init__(
            self,
            path_to_config: str = None,
    ) -> None:
        if path_to_config is None:
            raise Exception("The path to the configuration file cannot be None.")
        if not os.path.exists(path_to_config):
            raise Exception(f"Path '{path_to_config}' does not exist.")

        with open(path_to_config, "r") as f:
            config = json.load(f)

        if "path_to_storage" not in config.keys():
            raise Exception("The path to storage must be provided in the configuration.")
        if not os.path.exists(config["path_to_storage"]):
            raise Exception(f"Path '{config['path_to_storage']}' does not exist.")

        self.__dict__.update(config)

        self.begin_time = None
        self.end_time = None

    def get(
            self,
            key: str
    ) -> Any:
        """
        Returns the value of the specified key.

        Args:
            key (str):
                The key whose value is to be returned.

        Returns:
            Any:
                The value of the specified key.
        """

        return self.__dict__[key]

    def set(
            self,
            key: str,
            value: Any
    ) -> Any:
        """
        Sets the value of the specified key.

        Args:
            key (str):
                The key whose value is to be set.
            value (Any):
                The value to be set for the specified key.

        Returns:
            Any:
                The value of the specified key.
        """

        self.__dict__[key] = value

    def update(
            self,
            config: dict
    ) -> None:
        """
        Updates the configuration parameters with the specified dictionary.

        Args:
            config (dict):
                A dictionary containing the configuration parameters to be updated.
        """

        self.__dict__.update(config)

    def start_experiment(
            self,
            keys_for_naming: list = ()
    ) -> None:
        """
        Initializes the experiments by setting the seed and device.

        Args:
            keys_for_naming (list, optional):
                The keys to be used for naming the experiments. Defaults to ().
        """

        if self.begin_time is None:
            self.begin_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

            path_to_experiment = os.path.join(
                self.get("path_to_storage"),
                "_".join([self.__dict__[key] for key in keys_for_naming]) + "_" + self.begin_time
            )
            os.makedirs(path_to_experiment, exist_ok=True)

            paths = {
                "path_to_model": os.path.join(path_to_experiment, "model"),
                "path_to_tokenizer": os.path.join(path_to_experiment, "tokenizer"),
                "path_to_configuration": os.path.join(path_to_experiment, "configuration"),
                "path_to_logs": os.path.join(path_to_experiment, "logs")
            }
            for _, path in paths.items():
                os.makedirs(path, exist_ok=True)

            paths["path_to_experiment"] = path_to_experiment

            self.__dict__.update(paths)

    def end_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Ends the experiments by setting the end time.
        """

        if self.end_time is None:
            self.end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    @property
    def status(
            self
    ) -> str:
        """
        Returns the status of the experiments.

        Returns:
            str:
                The status of the experiments.
        """

        if self.begin_time is None:
            return "Not started"
        elif self.end_time is None:
            return "Running"
        else:
            return "Completed"
