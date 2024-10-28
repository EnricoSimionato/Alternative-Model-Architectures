import copy

import torch

import transformers


def get_parameters(
        module_tree: [torch.nn.Module | transformers.AutoModel],
        target_paths: list,
        layers_storage: {},
        blacklist: list = (),
        path: list = None,
        **kwargs
) -> None:
    """
    Extracts the matrices from the model tree.

    Args:
        module_tree ([torch.nn.Module | transformers.AutoModel]):
            The model tree.
        target_paths (list):
            The path of the targets.
        layers_storage (dict):
            Storage where the extracted layers will be at the end of the extraction.
        blacklist (list, optional):
            The list of blacklisted paths. Defaults to ().
        path (list, optional):
            The path to the current layer. Defaults to None.
    """

    for layer_name in module_tree._modules.keys():
        # Extracting the child from the current module
        child = module_tree._modules[layer_name]
        layer_path = copy.deepcopy(path) + [f"{layer_name}"] if path is not None else [f"{layer_name}"]

        if len(child._modules) == 0 and not isinstance(child, torch.nn.ModuleDict):
            target_paths_in_current_path = [
                is_subsequence(
                    [sub_path for sub_path in target_path if sub_path != "block_index"],
                    layer_path
                ) and not any(blacklisted_string in layer_path for blacklisted_string in blacklist)
                for target_path in target_paths]
            if sum(target_paths_in_current_path) > 1:
                raise Exception(f"The layer {layer_path} corresponds to multiple targets.")
            if any(target_paths_in_current_path):
                # Storing the layer in the dictionary of extracted layers
                if tuple(layer_path) in layers_storage.keys():
                    raise Exception(f"Layer {layer_path} already stored.")

                layers_storage[tuple(layer_path)] = child
        else:
            # Recursively calling the function
            get_parameters(
                module_tree=child,
                target_paths=target_paths,
                layers_storage=layers_storage,
                blacklist=blacklist,
                path=layer_path,
                **kwargs
            )

def is_subsequence(
        subsequence: list | tuple,
        sequence: list | tuple
) -> bool:
    """
    Checks if a sequence is a subsequence of another sequence.

    Args:
        subsequence (list | tuple):
            The subsequence.
        sequence (list | tuple):
            The sequence.

    Returns:
        bool:
            True if the subsequence is a subsequence of the sequence, False otherwise.
    """

    i = 0
    for element in sequence:
        if element == subsequence[i]:
            i += 1
        if i == len(subsequence):
            return True
    return False
