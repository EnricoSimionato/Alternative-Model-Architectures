from typing import Literal

from torchmetrics.classification import MulticlassStatScores


class ClassificationStats(MulticlassStatScores):
    """

    """

    def __init__(
            self,
            num_classes: int,
            top_k: int = 1,
            average: str = None,
            multidim_average: Literal["global", "samplewise"] = "global",
            ignore_index: int = None,
            validate_args: bool = True,
            **kwargs
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs
        )

    def get_stats(
            self
    ) -> tuple[float, float, float, float, float]:
        """

        """

        stats = self.compute()
        if self.num_classes == 2:
            print(stats)
            stats = stats[1, :]
            return stats[0], stats[1], stats[2], stats[3], stats[4]
        else:
            return stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3], stats[:, 4]

    def accuracy(
            self
    ) -> float:
        """

        """

        tp, fp, tn, fn, _ = self.get_stats()
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        if self.num_classes > 2:
            accuracy = accuracy.mean()

        return accuracy

    def precision(
            self
    ) -> float:
        """

        """

        tp, fp, tn, fn, _ = self.get_stats()
        precision = tp / (tp + fp)
        if self.num_classes > 2:
            precision = precision.mean()

        return precision

    def recall(
            self
    ) -> float:
        """

        """

        tp, fp, tn, fn, _ = self.get_stats()
        recall = tp / (tp + fn)
        if self.num_classes > 2:
            recall = recall.mean()

        return recall

    def f1_score(
            self
    ) -> float:
        """

        """

        tp, fp, tn, fn, _ = self.get_stats()
        precision = self.precision()
        recall = self.recall()
        f1_score = 2 * (precision * recall) / (precision + recall)
        if self.num_classes > 2:
            f1_score = f1_score.mean()

        return f1_score


if __name__ == "__main__":
    import torch

    test_stats = ClassificationStats(2)

    y_true = torch.tensor([0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 1, 1])

    test_stats.update(y_pred, y_true)

    print(test_stats.accuracy())
    print(test_stats.precision())
    print(test_stats.recall())
    print(test_stats.f1_score())
