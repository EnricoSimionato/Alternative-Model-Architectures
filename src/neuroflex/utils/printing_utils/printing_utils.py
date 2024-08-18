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
