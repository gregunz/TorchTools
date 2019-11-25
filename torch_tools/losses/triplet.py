import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss as Loss, MSELoss


def triplet_loss(anchor: Tensor, positives: Tensor, negatives: Tensor,
                 distance_fn=MSELoss(), margin=0.0, lambda_pos=1.0, lambda_neg=1.0):
    n_dim = anchor.dim()
    if positives.dim() == n_dim:
        positives = positives.unsqueeze(0)
    if negatives.dim() == n_dim:
        negatives = negatives.unsqueeze(0)

    loss = margin

    for pos in positives:
        dist_pos = distance_fn(anchor, pos)
        loss = lambda_pos * dist_pos + loss

    for neg in negatives:
        dist_neg = distance_fn(anchor, neg)
        loss = (-1) * lambda_neg * dist_neg + loss

    return torch.relu(loss)


class TripletLoss(Loss):
    def __init__(self, distance_fn, margin, lambda_pos=1.0, lambda_neg=1.0):
        super().__init__()
        self.distance_fn = distance_fn
        self.margin = margin
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def forward(self, anchor: Tensor, positives: Tensor, negatives: Tensor):
        return triplet_loss(
            anchor=anchor, positives=positives, negatives=negatives,
            distance_fn=self.distance_fn, margin=self.margin, lambda_pos=self.lambda_pos, lambda_neg=self.lambda_neg,
        )
