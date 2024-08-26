from enum import Enum


class Verbose(Enum):
    SILENT = 0
    INFO = 1
    DEBUG = 2

    def __lt__(self, other):
        if isinstance(other, Verbose):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Verbose):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Verbose):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Verbose):
            return self.value >= other.value
        return NotImplemented

    def print(
            self,
            message: str,
            verbosity_threshold: "Verbose"
    ) -> None:
        """
        Prints the message according to the verbosity level.

        Args:
            message (str):
                The message to be printed.
            verbosity_threshold (Verbose):
                The verbosity threshold to be compared with.
        """

        if self >= verbosity_threshold:
            print(f"{message}")
