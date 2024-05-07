from typing import Literal

import torch
from torchmetrics.classification import MulticlassStatScores


class ClassificationStats(MulticlassStatScores):
    """
    Class for computing classification statistics.

    Args:
        num_classes (int):
            The number of classes in the classification problem.
        top_k (int):
            Number of highest probability or logit score predictions considered to find the correct label. Only works
            when the predictions contain probabilities/logits.
        average (str):
            The method to average the statistics. For now, it is not used, it is just stored.
        multidim_average (str):
            The method to handle additionally dimensions ... . Should be one of the following:
                - global: Additional dimensions are flatted along the batch dimension
                - samplewise: Statistic will be calculated independently for each sample on the N axis. The statistics
                  in this case are calculated over the additional dimensions.
        ignore_index (int):
            Target value that is ignored and does not contribute to the metric calculation.
        validate_args (bool):
            Whether to validate the arguments and tensors should be validated for correctness. Set to False for faster
            computations.
        **kwargs:
            Additional arguments for the parent class.
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
            average=None,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs
        )
        self.type_of_average = average

    def get_stats(
            self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the statistics of the classification problem.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The true positives, false positives, true negatives, false negatives, and support statistics.
        """

        stats = self.compute()
        if self.num_classes == 2:
            stats = stats[1, :]
            return stats[0], stats[1], stats[2], stats[3], stats[4]
        else:
            return stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3], stats[:, 4]

    def accuracy(
            self
    ) -> torch.Tensor:
        """
        Computes the accuracy given the current statistics of the model.

        Returns:
            torch.Tensor:
                The accuracy of the model.
        """

        tp, fp, tn, fn, _ = self.get_stats()
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        if self.num_classes > 2:
            accuracy = accuracy.mean()

        return accuracy

    def precision(
            self
    ) -> torch.Tensor:
        """
        Computes the precision given the current statistics of the model.

        Returns:
            torch.Tensor:
                The precision of the model.
        """

        tp, fp, tn, fn, _ = self.get_stats()
        precision = tp / (tp + fp)
        if self.num_classes > 2:
            precision = precision.mean()

        return precision

    def recall(
            self
    ) -> torch.Tensor:
        """
        Computes the recall given the current statistics of the model.

        Returns:
            torch.Tensor:
                The recall of the model.
        """

        tp, fp, tn, fn, _ = self.get_stats()
        recall = tp / (tp + fn)
        if self.num_classes > 2:
            recall = recall.mean()

        return recall

    def f1_score(
            self
    ) -> torch.Tensor:
        """
        Computes the F1 score given the current statistics of the model.

        Returns:
            torch.Tensor:
                The F1 score of the model.
        """

        precision = self.precision()
        recall = self.recall()
        f1_score = 2 * (precision * recall) / (precision + recall)
        if self.num_classes > 2:
            f1_score = f1_score.mean()

        return f1_score


class TestClassificationStats:
    """
    Class for testing the ClassificationStats class.
    """

    def test_classification_stats(
            self
    ) -> None:
        """
        Tests the ClassificationStats class.
        """

        test_stats = ClassificationStats(2)

        y_true = torch.tensor([0, 1, 1, 0])
        y_pred = torch.tensor([0, 1, 1, 1])

        test_stats.update(y_pred, y_true)

        assert test_stats.accuracy() == 0.75
        assert test_stats.precision() == 0.6666666666666666
        assert test_stats.recall() == 1.0
        assert test_stats.f1_score() == 0.8
