import torchmetrics


class ClassificationStats(torchmetrics.classification.MulticlassStatScores):
    def __init___(
            self,
            num_classes: int,
            average: str = None,
            #mdmc_average: str = "global",
            #compute_on_step: bool = True,
            #dist_sync_on_step: bool = False,
            #process_group=None,
            #dist_sync_fn=None,
    ) -> None:
        super().__init__(num_classes=num_classes, average=average)

    def get_stats(
            self
    ):
        stats = self.compute()
        if self.num_classes == 2:
            stats = stats[1, :]
            return stats[0], stats[1], stats[2], stats[3], stats[4]
        else:
            return stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3], stats[:, 4]

    def accuracy(self):
        tp, fp, tn, fn, _ = self.get_stats()
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        if self.num_classes > 2:
            accuracy = accuracy.mean()
        return accuracy

    def precision(self):
        tp, fp, tn, fn, _ = self.get_stats()
        precision = tp / (tp + fp)
        if self.num_classes > 2:
            precision = precision.mean()
        return precision

    def recall(self):
        tp, fp, tn, fn, _ = self.get_stats()
        recall = tp / (tp + fn)
        if self.num_classes > 2:
            recall = recall.mean()
        return recall

    def f1_score(self):
        tp, fp, tn, fn, _ = self.get_stats()
        precision = self.precision()
        recall = self.recall()
        f1_score = 2 * (precision * recall) / (precision + recall)
        if self.num_classes > 2:
            f1_score = f1_score.mean()
        return f1_score


if __name__ == "__main__":
    import torch

    test_stats = ClassificationStats(2, average=None)

    y_true = torch.tensor([0, .8, 1, 0])
    y_pred = torch.tensor([0, 1, 1, 1])

    test_stats.update(y_pred, y_true)

    print(test_stats.accuracy())
    print(test_stats.precision())
    print(test_stats.recall())
    print(test_stats.f1_score())
