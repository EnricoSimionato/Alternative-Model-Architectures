
def convert_seconds_in_other_units(
        number_of_seconds: float,
        units_of_measure: list = ("h", "m", "s")
) -> dict:
    """
    Converts a given number of seconds into other units of measure.

    Args:
        number_of_seconds (float):
            The number of seconds to be converted.
        units_of_measure (list, optional):
            List of units of measure into which the bytes are to be converted.
            Defaults to ["h", "m", "s"]. Supported units are "s" (seconds), "m" (minutes), "h" (hours), "d" (days).

    Returns:
        dict:
            A dictionary containing the converted value in each specified unit of measure.

    Raises:
        Exception:
            If an invalid unit of measure is provided.

    Note:
        The function returns the conversion in the specified units of measure
        while maintaining precision up to the second last specified unit. The
        final unit will have the remaining bytes after conversion.
    """

    units_of_measure = [unit.lower() for unit in units_of_measure]
    valid_units_of_measure = {
        "d": 60 * 60 * 24,
        "h": 60 * 60,
        "m": 60,
        "s": 1
    }

    for unit in units_of_measure:
        if unit not in valid_units_of_measure.keys():
            raise Exception(f"Invalid units of measure for the time, the valid ones are: {valid_units_of_measure.keys()}")

    units_of_measure_to_use = {}
    for unit in valid_units_of_measure.keys():
        if unit in units_of_measure:
            units_of_measure_to_use[unit] = valid_units_of_measure[unit]

    if len(units_of_measure_to_use) <= 0:
        units_of_measure_to_use["s"] = 0

    conversion = {}
    current_number_of_bytes = number_of_seconds
    for unit in list(units_of_measure_to_use.keys())[:-1]:
        conversion[unit] = current_number_of_bytes // (units_of_measure_to_use[unit])
        current_number_of_bytes -= conversion[unit] * (units_of_measure_to_use[unit])

    conversion[list(units_of_measure_to_use.keys())[-1]] = current_number_of_bytes / (units_of_measure_to_use[list(units_of_measure_to_use.keys())[-1]])

    return conversion
