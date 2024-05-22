from abc import ABC, abstractmethod

from gbm import GlobalDependentLinear, GlobalDependentEmbedding
from gbm.layers.global_dependent_layer import GlobalDependent



class GlobalDependentAverageMatrix(GlobalDependent, ABC):
    def __init__(self, num_classes, num_features, num_groups, num_layers):
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_groups = num_groups
        self.num_layers = num_layers

    @abstractmethod
    def post_initialize(
            self
    ) -> None:
        """

        """

class GlobalAverageLinear(GlobalDependentLinear, GlobalDependentAverageMatrix):
    def __init__(
            self
    ) -> None:
        super().__init__()


class GlobalAverageEmbedding(GlobalDependentEmbedding, GlobalDependentAverageMatrix):
    def __init__(
            self
    ) -> None:
        super().__init__()
