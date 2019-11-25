from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss


def triplet_w2v_loss(anchor: Tensor, positives: Tensor, negatives: Tensor, margin=0.0, lambda_pos=1.0, lambda_neg=1.0):
    n_dim = anchor.dim()
    assert n_dim == 3, f'expecting 3 dimensions for anchor and not {n_dim}'

    if positives.dim() == n_dim:
        positives = positives.unsqueeze(0)
    if negatives.dim() == n_dim:
        negatives = negatives.unsqueeze(0)

    loss = margin

    anchor = anchor.transpose(1, 2)
    for pos in positives:
        dist_pos = (-1) * F.logsigmoid(anchor.bmm(pos))
        loss = lambda_pos * dist_pos + loss

    for neg in negatives:
        dist_neg = (-1) * F.logsigmoid((-1) * anchor.bmm(neg))
        loss = lambda_neg * dist_neg + loss

    return loss


class TripletW2VLoss(Loss):
    def __init__(self, distance_fn, margin, lambda_pos=1.0, lambda_neg=1.0):
        super().__init__()
        self.distance_fn = distance_fn
        self.margin = margin
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def forward(self, anchor: Tensor, positives: Tensor, negatives: Tensor):
        return triplet_w2v_loss(
            anchor=anchor, positives=positives, negatives=negatives,
            margin=self.margin, lambda_pos=self.lambda_pos, lambda_neg=self.lambda_neg,
        )
