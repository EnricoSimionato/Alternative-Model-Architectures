import os
from datetime import datetime
import json
from typing import Any

from enum import Enum

import transformers


class ExperimentStatus(Enum):
    NOT_STARTED = "Not started"
    RUNNING = "Running"
    COMPLETED = "Completed"


class Config:
    """
    Config class to store all the configuration parameters of an experiment about training a deep model using Pytorch
    Lightning.

    Args:
        path_to_config (str):
            The path to the configuration file.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        All the keys in the config file are added as attributes to the class.
        keys_for_naming (list):
            A list of keys to be used for the naming of the experiment. The values of these keys will be concatenated
            to form the name of the experiment. If this is not provided in the configuration file, the name of the
            experiment will be the date and time when the experiment started.
        begin_time (str):
            The time when the experiment started. It is set to None initially and is updated when the experiment starts.
        end_time (str):
            The time when the experiment ended. It is set to None initially and is updated when the experiment ends.
    """

    def __init__(
            self,
            path_to_config: str = None,
            **kwargs
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

        self.keys_for_naming = []
        if "keys_for_naming" in config.keys():
            for key in config["keys_for_naming"]:
                if key not in config.keys():
                    raise Exception(f"The key '{key}', which is one the keys to be used for the naming of the "
                                    f"experiment must be provided in the configuration.")

        self.__dict__.update(config)

        self.begin_time = None
        self.end_time = None

    def contains(
            self,
            key: str,
            **kwargs
    ) -> bool:
        """
        Checks if the specified key is present in the configuration.

        Args:
            key (str):
                The key to be checked.
            **kwargs:
                Additional keyword arguments.

        Returns:
            bool:
                True if the key is present, False otherwise.
        """

        return key in self.__dict__.keys()

    def get(
            self,
            key: str,
            **kwargs
    ) -> Any:
        """
        Returns the value of the specified key.

        Args:
            key (str):
                The key whose value is to be returned.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Any:
                The value of the specified key.
        """

        return self.__dict__[key]

    def get_paths(
            self,
            **kwargs
    ) -> dict:
        """
        Returns the paths of the experiment.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the paths of the experiment.
        """

        paths = {}

        for key in self.__dict__.keys():
            if "path" in key:
                paths[key] = self.__dict__[key]

        return paths

    def get_original_model(
            self
    ) -> transformers.PreTrainedModel:
        """
        Returns the original model.

        Returns:
            transformers.PreTrainedModel:
                The original model.
        """

        if self.contains("task"):
            if self.get("task") == "chatbot":
                original_model = transformers.AutoModelForCausalLM.from_pretrained(self.get("path_to_model"))
            elif self.get("task") == "classification":
                original_model = transformers.AutoModelForSequenceClassification.from_pretrained(self.get("path_to_model"))
            else:
                original_model = transformers.AutoModel.from_pretrained(self.get("path_to_model"))
                print("Configuration does not contain a know task, loading the model as a generic model.")
                print(f"The given task is {self.get('task')}")
        else:
            original_model = transformers.AutoModel.from_pretrained(self.get("path_to_model"))
            print("Configuration does not contain the task, loading the model as a generic model.")

        return original_model

    def set(
            self,
            key: str,
            value: Any,
            **kwargs
    ):
        """
        Sets the value of the specified key.

        Args:
            key (str):
                The key whose value is to be set.
            value (Any):
                The value to be set for the specified key.
            **kwargs:
                Additional keyword arguments.
        """

        self.__dict__[key] = value

    def update(
            self,
            config: dict,
            **kwargs
    ) -> None:
        """
        Updates or inserts the configuration parameters according to the provided dictionary.

        Args:
            config (dict):
                A dictionary containing the configuration parameters to be updated or inserted.
            **kwargs:
                Additional keyword arguments.
        """

        self.__dict__.update(config)

    def start_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Initializes the experiments by defining the paths to the directories of the experiment and the start timestamp
        of the experiment.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        if self.begin_time is None:
            self.begin_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

            dir_name = "_".join([self.__dict__[key].replace("/", "_") for key in self.keys_for_naming])
            path_to_experiment = os.path.join(
                self.get("path_to_storage"),
                dir_name +
                ("_" if len(dir_name) > 0 else "") +
                self.begin_time
            )
            os.makedirs(path_to_experiment, exist_ok=True)

            paths = {
                "path_to_model": os.path.join(path_to_experiment, "model"),
                "path_to_tokenizer": os.path.join(path_to_experiment, "tokenizer"),
                "path_to_configuration": os.path.join(path_to_experiment, "configuration"),
                "path_to_logs": os.path.join(path_to_experiment, "logs"),
                #"path_to_tensorboard_logs": os.path.join(path_to_experiment, "logs", "tensorboard_logs"),
                #"path_to_csv_logs": os.path.join(path_to_experiment, "logs", "csv_logs"),
                "path_to_checkpoints": os.path.join(path_to_experiment, "checkpoints")
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

        Args:
            kwargs:
                Additional keyword arguments.
        """

        if self.end_time is None:
            self.end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    @property
    def status(
            self
    ) -> ExperimentStatus:
        """
        Returns the status of the experiments.

        Returns:
            ExperimentStatus:
                The status of the experiments.
        """

        if self.begin_time is None:
            return ExperimentStatus.NOT_STARTED
        elif self.end_time is None:
            return ExperimentStatus.RUNNING
        else:
            return ExperimentStatus.COMPLETED

    def show_config(
            self
    ) -> None:
        import pandas as pd
        config_df = pd.DataFrame(list(self.__dict__.items()), columns=['Attribute', 'Value'])

        return config_df

    def __str__(
            self
    ) -> str:
        """
        Returns the string representation of the configuration.

        Returns:
            str:
                The string representation of the configuration.
        """

        config_string = ""
        for key in self.__dict__.keys():
            config_string += f"{key}: {self.__dict__[key]}\n"
        return config_string
