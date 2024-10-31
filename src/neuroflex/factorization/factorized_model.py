from abc import ABC

import torch


class FactorizedModel(ABC, torch.nn.Module):
    """
    Model with factorized layers.

    Args:
        target_model (PreTrainedModel):
            Pretrained model.
        target_layers (dict):
            Layers to factorize.
        use_names_as_keys (bool):
            Whether to use the names of the layers in the keys of the global layers, having different global layers
            for layers having different roles in the original model.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers. Allowing to group layers with different
            names to have the same global layer.
        remove_average (bool):
            Whether to remove the average matrices from the layers of the model. Averages are computed considering the
            grouping imposed by target_layers or mapping_layer_name_key.
        from_pretrained (bool):
            Whether the model is being loaded from a pretrained model.
        preserve_original_model (bool):
            Whether to preserve the target model or to change directly it.
        verbose (int):
            Verbosity level.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        target_layers (dict):
            Layers to factorize.
        mapping_layer_name_key (dict):
            Mapping of the layer names to the keys of the global layers.
        model (PreTrainedModel):
            Pretrained model.
        global_layers (torch.nn.ModuleDict):
            Global layers.
        conversions (dict):
            Mapping of layer types to global-dependent layer classes.
        info (dict):
            Information about the model.
        verbose (Verbose):
            Verbosity level.
    """