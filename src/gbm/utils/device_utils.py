from __future__ import annotations

import torch


def get_available_device(
    preferred_device: str = "cuda",
    just_string: bool = False
) -> torch.Device | str:
    """
    Retrieves the available device.

    Args:
        preferred_device (str):
            The preferred device to use.
        just_string (bool):
            Whether to return the device as a string. Defaults to False.

    Returns:
        torch.Device | str:
            The available device.
    """

    available_devices = {
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": torch.device("cuda")
        },
        "mps": {
            "available": torch.backends.mps.is_available(),
            "device": torch.device("mps")
        },
        "cpu": {
            "available": True,
            "device": torch.device("cpu")
        }
    }

    if preferred_device in available_devices.keys() and available_devices[preferred_device]["available"]:
        if just_string:
            return preferred_device
        return available_devices[preferred_device]["device"]

    for device in available_devices.keys():
        if available_devices[device]["available"]:
            if just_string:
                return device
            return available_devices[device]["device"]
