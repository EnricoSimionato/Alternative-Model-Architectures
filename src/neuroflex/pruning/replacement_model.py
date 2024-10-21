import copy
from typing import override

import torch

import transformers

from exporch import Config

from redhunter import ProcessedLayerReplacingModelWrapper


class SharedAverageLayerReplacingModelWrapper(ProcessedLayerReplacingModelWrapper):
    """
    Class to replace layers in a model with a shared average layer.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model to be wrapped.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them. The source_layer_path will be ignored, if given.

    Attributes:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The wrapped model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them.
            The structure of the dictionary is as follows:
            {
                [destination_layer_path_1]: [source_layer_path_1],
                [destination_layer_path_2]: [source_layer_path_2],
                ...

            }
            The source_layer_path will be ignored, if given.
    """

    def __init__(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel],
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple] = None,
    ) -> None:
        super().__init__(
            model,
            destination_layer_path_source_layer_path_mapping
        )

    @override
    def preprocess_source_layers(
            self,
            source_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module]
    ) -> dict[list | tuple: torch.nn.Module]:
        """
        Pre-processes the source layers.

        Args:
            source_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.

        Returns:
            dict[str: torch.nn.Module]:
                The pre-processed source layers.
        """

        print("Preprocessing the source layers.")
        source_layer_path_average_layer_mapping = {}

        source_layer_path_source_layer_grouped_mapping = self.group_layers(source_layer_path_source_layer_mapping)
        for source_layer_paths, source_layers_to_average in source_layer_path_source_layer_grouped_mapping.items():
            average_layer = self.compute_average_layer(source_layers_to_average)
            for source_layer_path in source_layer_paths:
                source_layer_path_average_layer_mapping.update({source_layer_path: average_layer})
        print("Preprocessing done.")
        return source_layer_path_average_layer_mapping

    @staticmethod
    def group_layers(
            source_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module]
    ) -> dict[list | tuple: list[torch.nn.Module]]:
        """
        Groups the source layers to be averaged.
        The grouping is done on a name basis without considering the block index.
        The method can be overridden to group the layers based on a different criterion.

        Args:
            source_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.

        Returns:
            dict[str: list[torch.nn.Module]]:
                The mapping between the paths of the source layers in the same group and their layers that have to be
                averaged.
        """

        source_layer_path_source_layer_grouped_mapping = {}

        source_layer_path_source_layer_grouped_mapping_utility = {}
        for source_layer_path, source_layer in source_layer_path_source_layer_mapping.items():
            filtered_source_layer_path = tuple([el for el in source_layer_path if not el.isdigit()])
            if filtered_source_layer_path not in source_layer_path_source_layer_grouped_mapping_utility:
                source_layer_path_source_layer_grouped_mapping_utility[filtered_source_layer_path] = [[], []]
            source_layer_path_source_layer_grouped_mapping_utility[filtered_source_layer_path][0].append(
                source_layer_path)
            source_layer_path_source_layer_grouped_mapping_utility[filtered_source_layer_path][1].append(
                source_layer)

        for source_layer_paths, source_layers in source_layer_path_source_layer_grouped_mapping_utility.values():
            source_layer_path_source_layer_grouped_mapping.update({tuple(source_layer_paths): source_layers})

        return source_layer_path_source_layer_grouped_mapping

    @staticmethod
    def compute_average_layer(
            layers_to_average: list[torch.nn.Module]
    ) -> torch.nn.Module:
        """
        Computes the average layer from the given layers.

        Args:
            layers_to_average (list[torch.nn.Module]):
                The layers to be averaged.

        Returns:
            torch.nn.Module:
                The average layer.
        """

        print("Computing the average layer.")
        average_layer = copy.deepcopy(layers_to_average[0])
        try:
            weight = torch.mean(torch.stack([layer.weight.data for layer in layers_to_average]), dim=0)
            print(weight)
            average_layer.weight = torch.nn.Parameter(weight)
            print(average_layer.weight)
        except AttributeError as e:
            print(f"Error computing the average layer weight: {e}")
            raise e
        try:
            bias = torch.mean(torch.stack([layer.bias.data for layer in layers_to_average]), dim=0)
            average_layer.bias = torch.nn.Parameter(bias)
            print(average_layer.bias)
        except AttributeError as e:
            print(f"Error computing the average layer bias: {e}")
            print("Setting the layer to have no bias.")
            average_layer.bias = None
        print("Average layer computed.")

        return average_layer


def get_layer_replaced_model(
        model: transformers.AutoModel,
        replacement_method: str,
        destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple],
        config: Config
) -> ProcessedLayerReplacingModelWrapper:
    """
    Returns the model with the replaced layers based on the given replacement method.

    Args:
        model (transformers.AutoModel):
            The original model to modify.
        replacement_method (str):
            The method to use to replace the layers.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them. The meaning of the paths depends on the replacement method.
        config (Config):
            The configuration parameters for the experiment.

    Returns:
        gbm.GlobalDependentModel:
            The factorized model to use in the experiment.
    """

    replacement_method = replacement_method.lower()

    if replacement_method == "sharedaveragelayer":
        layer_replaced_model = SharedAverageLayerReplacingModelWrapper(
            model,
            destination_layer_path_source_layer_path_mapping
        )
    else:
        raise ValueError("Replacement method not recognized")

    return layer_replaced_model
