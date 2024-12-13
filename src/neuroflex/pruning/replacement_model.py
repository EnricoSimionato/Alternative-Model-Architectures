import copy
from typing import override

import torch

import transformers

from exporch import Config

from redhunter import ProcessedLayerReplacingModelWrapper


class SharedAverageLayerReplacingModelWrapper(ProcessedLayerReplacingModelWrapper):
    """
    Class to replace layers in a model with a shared average layer.

    Attributes:
        model (transformers.PreTrainedModel | transformers.AutoModel):
            The model to wrap.
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

        Args:
            model (transformers.PreTrainedModel | transformers.AutoModel):
                The model to wrap.
            destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
                The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
                them.
                The source_layer_path will be ignored, if given.
    """

    def __init__(
            self,
            model: transformers.PreTrainedModel | transformers.AutoModel,
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple] = None,
    ) -> None:
        super().__init__(model, destination_layer_path_source_layer_path_mapping)

    @override
    def preprocess_source_layers(
            self,
            source_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module]
    ) -> dict[list | tuple: torch.nn.Module]:
        """
        Pre-processes the source layers.
        The method groups the layers based on their name and averages them.

        Args:
            source_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.

        Returns:
            dict[str: torch.nn.Module]:
                The mapping between the path to the source layers and the associated pre-processed weights.
        """

        source_layer_path_average_layer_mapping = {}
        source_layer_path_source_layer_grouped_mapping = self.group_layers(source_layer_path_source_layer_mapping)
        for source_layer_paths, source_layers_to_average in source_layer_path_source_layer_grouped_mapping.items():
            average_layer = self.compute_average_layer(source_layers_to_average)

            for source_layer_path in source_layer_paths:
                source_layer_path_average_layer_mapping.update({source_layer_path: average_layer})

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
        Given some layers, they are traversed recursively, and the weights and biases of each base module, e.g.
        torch.nn.Linear, are averaged.

        Args:
            layers_to_average (list[torch.nn.Module]):
                The layers to average.

        Returns:
            torch.nn.Module:
                The average layer.
        """

        if not layers_to_average:
            raise ValueError("The blocks list is empty.")

        # Copying the structure of the first block
        average_block = copy.deepcopy(layers_to_average[0])

        def average_parameters(modules: list[torch.nn.Module]) -> dict[str: torch.Tensor]:
            """"
            Recursively averages weights and biases of submodules.

            Args:
                modules (list[torch.nn.Module]):
                    The modules to average.

            Returns:
                dict[str: torch.Tensor]:
                    The averaged parameters.
            """

            # Collecting parameter names and their aggregated data
            param_sums = {}
            dtypes = {}
            count = len(modules)

            for module in modules:
                for name, param in module.named_parameters(recurse=False):
                    if param is not None:
                        if name not in param_sums:
                            param_sums[name] = param.data.clone().to(torch.float32)
                            dtypes[name] = param.data.dtype
                        else:
                            param_sums[name] += param.data.to(torch.float32)

            # Computing the average
            averaged_params = {name: param.to(dtypes[name]) / count for name, param in param_sums.items()}
            return averaged_params

        def apply_averaged_parameters(target_module: torch.nn.Module, averaged_params: dict[str: torch.Tensor]) -> None:
            """
            Applies the averaged parameters to the target module.

            Args:
                target_module (torch.nn.Module):
                    The target module to apply the averaged parameters to.
                averaged_params (dict[str: torch.Tensor]):
                    The averaged parameters to apply.
            """

            for name, param in target_module.named_parameters(recurse=False):
                if name in averaged_params:
                    param.data.copy_(averaged_params[name])

        def recursive_average(source_blocks: list[torch.nn.Module], target_block: torch.nn.Module) -> None:
            """
            Recursively traverses submodules for averaging.

            Args:
                source_blocks (list[torch.nn.Module]):
                    Source layers to average.
                target_block (torch.nn.Module):
                    Target layer to store the averaged parameters.
            """
            # Base Case: If the module itself has parameters, average them directly
            if list(target_block.parameters(recurse=False)):
                averaged_params = average_parameters(source_blocks)
                apply_averaged_parameters(target_block, averaged_params)
                return  # No need to recurse further

            # Recursive Case: Traverse submodules
            for name, submodule in target_block.named_children():
                # Collecting submodules with the same name across blocks
                submodules_to_average = [getattr(block, name) for block in source_blocks]

                # Recursing into nested submodules
                recursive_average(submodules_to_average, getattr(target_block, name))
        # Starting recursive averaging
        recursive_average(layers_to_average, average_block)

        return average_block

def get_layer_replaced_model(
        model: transformers.AutoModel | transformers.PreTrainedModel,
        replacement_method: str,
        destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple],
        config: Config
) -> ProcessedLayerReplacingModelWrapper:
    """
    Returns the model with the replaced layers based on the chosen replacement method.

    Args:
        model (transformers.AutoModel | transformers.PreTrainedModel):
            The original model to wrap.
        replacement_method (str):
            The method to use to replace the layers.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them.
            The meaning of the paths depends on the replacement method.
        config (Config):
            The configuration parameters for the experiment.

    Returns:
        gbm.GlobalDependentModel:
            The factorized model to use in the experiment.
    """

    replacement_method = replacement_method.lower()

    if replacement_method == "sharedaveragelayer":
        layer_replaced_model = SharedAverageLayerReplacingModelWrapper(
            model, destination_layer_path_source_layer_path_mapping
        )
    else:
        raise ValueError("Replacement method not recognized")

    return layer_replaced_model
