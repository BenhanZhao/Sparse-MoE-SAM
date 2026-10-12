import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    """Table 8: stable BCE + soft Dice; model supplies balance formulation.

    Post-softmax attention L1 is constant (=1 per query). We expose the paper
    term faithfully, but do not pretend it promotes additional sparsity.
    Optional negative entropy promotes diverse attention/gating (Sec. 4.2/4.3).
    """
    def __init__(self, balance_weight=.01, sparse_weight=.001, entropy_weight=0.,
                 pos_weight=None):
        super().__init__()
        self.balance_weight, self.sparse_weight, self.entropy_weight = balance_weight, sparse_weight, entropy_weight
        self.register_buffer("pos_weight", torch.tensor(pos_weight) if pos_weight is not None else None)

    def forward(self, logits, target, aux=None, regularize=True):
        if logits.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), pos_weight=self.pos_weight)
        probabilities = logits.float().sigmoid()
        dimensions = tuple(range(1, logits.ndim))
        dice = 1 - ((2 * (probabilities * target).sum(dimensions) + 1e-6) /
                    (probabilities.sum(dimensions) + target.sum(dimensions) + 1e-6)).mean()
        total = bce + dice
        if aux is not None and regularize:
            total = total + self.balance_weight * aux["balance"] + self.sparse_weight * aux["sparse"]
            total = total - self.entropy_weight * (aux["attention_entropy"] + aux["gate_entropy"])
        return total


class BinaryMetrics:
    """Dataset micro confusion counts; all-background agreement scores 1.

    Foreground IoU/Dice and mean foreground/background IoU are separate.
    """
    def __init__(self, threshold=.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits, target):
        prediction, truth = logits.detach().sigmoid() >= self.threshold, target >= .5
        self.tp += int((prediction & truth).sum())
        self.fp += int((prediction & ~truth).sum())
        self.fn += int((~prediction & truth).sum())
        self.tn += int((~prediction & ~truth).sum())

    def compute(self):
        ratio = lambda numerator, denominator: numerator / denominator if denominator else 1.0
        iou = ratio(self.tp, self.tp + self.fp + self.fn)
        background = ratio(self.tn, self.tn + self.fp + self.fn)
        return {"iou": iou, "miou": (iou + background) / 2,
                "dice": ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
                "precision": ratio(self.tp, self.tp + self.fp),
                "recall": ratio(self.tp, self.tp + self.fn)}
