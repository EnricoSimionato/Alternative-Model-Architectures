from __future__ import annotations

import transformers


def convert_bytes_in_other_units(
        number_of_bytes: float,
        units_of_measure: list = ("GB", "MB")
) -> dict:
    """
    Converts a given number of bytes into other units of measure.

    Args:
        number_of_bytes (float):
            The number of bytes to be converted.
        units_of_measure (list, optional):
            List of units of measure into which the bytes are to be converted.
            Supported units are "B" (bytes), "KB" (kilobytes), "MB" (megabytes), "GB" (gigabytes), and "TB" (terabytes).

    Returns:
        dict:
            A dictionary containing the converted value in each specified unit of measure.

    Raises:
        Exception:
            If an invalid unit of measure is provided.

    Note:
        The function returns the conversion in the specified units of measure while maintaining precision up to the
        second last specified unit.
        The final unit will have the remaining bytes after conversion.
    """

    units_of_measure = [unit.upper() + "B" if unit.upper()[-1] != "B" else unit.upper() for unit in units_of_measure]
    valid_units_of_measure = {
        "TB": 4,
        "GB": 3,
        "MB": 2,
        "KB": 1,
        "B": 0
        }

    for unit in units_of_measure:
        if unit not in valid_units_of_measure.keys():
            raise Exception(f"Invalid units of measure for the memory, the valid ones are: {valid_units_of_measure.keys()}")

    units_of_measure_to_use = {}
    for unit in valid_units_of_measure.keys():
        if unit in units_of_measure:
            units_of_measure_to_use[unit] = valid_units_of_measure[unit]

    if len(units_of_measure_to_use) <= 0:
        units_of_measure_to_use["B"] = 0

    conversion = {}
    current_number_of_bytes = number_of_bytes
    for unit in list(units_of_measure_to_use.keys())[:-1]:
        conversion[unit] = current_number_of_bytes // (1024 ** units_of_measure_to_use[unit])
        current_number_of_bytes -= conversion[unit] * (1024 ** units_of_measure_to_use[unit])

    conversion[list(units_of_measure_to_use.keys())[-1]] =  current_number_of_bytes / (1024 ** units_of_measure_to_use[list(units_of_measure_to_use.keys())[-1]])

    return conversion


def compute_model_memory_usage(
        model: transformers.PreTrainedModel | transformers.AutoModel,
        unit_of_measure: str = "B"
) -> float:
    """
    Computes the memory usage of a PyTorch model in the specified unit of measure.

    Args:
        model (transformers.PreTrainedModel | transformers.AutoModel):
            The PyTorch model whose memory usage is to be computed.
        unit_of_measure (str, optional):
            The unit of measure for memory usage. Defaults to "B".
            Supported units are "B" (bytes), "KB" (kilobytes), "MB" (megabytes), "GB" (gigabytes), and "TB" (terabytes).

    Returns:
        float:
            The memory usage of the model in the specified unit of measure.

    Note:
        The memory usage is calculated based on the sizes of the model parameters.
    """

    unit_of_measure = unit_of_measure.upper()

    bytes_usage = 0
    for param in model.parameters():
        bytes_usage += param.numel() * param.element_size()

    power = 0
    if unit_of_measure in ["KB", "K"]:
        power = 1
    elif unit_of_measure in ["MB", "M"]:
        power = 2
    elif unit_of_measure in ["GB", "G"]:
        power = 3
    elif unit_of_measure in ["TB", "T"]:
        power = 4
    else:
        print("Unknown unit of measure, B, KB, MB, GB, and TB are allowed.")
        print("Returning memory usage in number of bytes.")

    memory_usage = bytes_usage / (1024 ** power)

    return memory_usage
