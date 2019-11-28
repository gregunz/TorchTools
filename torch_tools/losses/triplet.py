import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import _Loss as Loss


def triplet_loss(anchor: Tensor, positives: Tensor, negatives: Tensor,
                 distance_fn=MSELoss(), margin=0.0, lambda_pos=1.0, lambda_neg=1.0):
    n_dim = anchor.dim()
    if positives.dim() == n_dim:
        positives = positives.unsqueeze(1)
    if negatives.dim() == n_dim:
        negatives = negatives.unsqueeze(1)

    loss = margin

    for i in range(positives.size(1)):
        dist_pos = distance_fn(anchor, positives[:, i])
        loss = lambda_pos * dist_pos + loss

    for i in range(negatives.size(1)):
        dist_neg = distance_fn(anchor, negatives[:, i])
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


def triplet_w2v_loss(anchor: Tensor, positives: Tensor, negatives: Tensor,
                     margin=0.0, lambda_pos=1.0, lambda_neg=1.0, reduction='mean'):
    # anchor = anchor.squeeze()
    # positives = positives.squeeze()
    # negatives = negatives.squeeze()

    n_batch = anchor.size(0)
    n_dim = anchor.dim()
    assert n_dim == 2, f'expecting 2 dimensions for anchor (tensor size = {anchor.size()})'

    if positives.dim() == n_dim:
        positives = positives.unsqueeze(1)
    if negatives.dim() == n_dim:
        negatives = negatives.unsqueeze(1)

    anchor = anchor.unsqueeze(1)  # matrix transposed (1st dim is batch)
    vec_dim = anchor.size(2)

    def compute_distance(pos_or_neg):
        n = pos_or_neg.size(1)
        anchor_exp = anchor.expand(n_batch, n, vec_dim).reshape(-1, 1, vec_dim)
        return anchor_exp.bmm(pos_or_neg.view(-1, vec_dim, 1))

    def reduce(loss):
        if reduction == 'mean':
            return torch.mean(loss.view(n_batch, -1), 0).sum()
        elif reduction == 'sum':
            return torch.sum(loss.view(n_batch, -1), 0).sum()
        else:
            raise NotImplementedError(reduction)

    loss_pos = (-1) * F.logsigmoid(compute_distance(positives))
    loss_pos = reduce(loss_pos) * lambda_pos

    loss_neg = (-1) * F.logsigmoid((-1) * compute_distance(negatives))
    loss_neg = reduce(loss_neg) * lambda_neg

    return loss_pos + loss_neg + margin


class TripletW2VLoss(TripletLoss):
    def forward(self, anchor: Tensor, positives: Tensor, negatives: Tensor):
        return triplet_w2v_loss(
            anchor=anchor, positives=positives, negatives=negatives,
            margin=self.margin, lambda_pos=self.lambda_pos, lambda_neg=self.lambda_neg,
        )
