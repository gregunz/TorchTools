import torch
from torch import Tensor
from torch.nn import functional as F


def triplet_loss(anchor: Tensor, positives: Tensor, negatives: Tensor,
                 margin=0.0, lambda_pos=None, lambda_neg=None, pos_neg_dist=False):
    n_dim = anchor.dim()
    if positives.dim() == n_dim:
        positives = positives.unsqueeze(1)
    if negatives.dim() == n_dim:
        negatives = negatives.unsqueeze(1)

    n_pos = positives.size(1)
    n_neg = negatives.size(1)

    if lambda_pos is None:
        lambda_pos = 1.0 / n_pos
    if lambda_neg is None:
        lambda_neg = 1.0 / n_neg
        if pos_neg_dist:
            lambda_neg /= n_pos

    loss = 0

    for i in range(n_pos):
        dist_pos = (anchor - positives[:, i]).pow(2).sum(1)
        loss += lambda_pos * dist_pos

    for i in range(n_neg):
        dist_neg = (anchor - negatives[:, i]).pow(2).sum(1)
        loss -= lambda_neg * dist_neg
        if pos_neg_dist:
            for j in range(positives.size(1)):
                dist_pos_neg = (positives[:, j] - negatives[:, i]).pow(2).sum(1)
                loss -= lambda_neg * dist_pos_neg

    return torch.relu(margin + loss).sum()


# class TripletLoss(Loss):
#     def __init__(self, distance_fn, margin, lambda_pos=1.0, lambda_neg=1.0):
#         super().__init__()
#         self.distance_fn = distance_fn
#         self.margin = margin
#         self.lambda_pos = lambda_pos
#         self.lambda_neg = lambda_neg
#
#     def forward(self, anchor: Tensor, positives: Tensor, negatives: Tensor):
#         return triplet_loss(
#             anchor=anchor, positives=positives, negatives=negatives,
#             distance_fn=self.distance_fn, margin=self.margin, lambda_pos=self.lambda_pos, lambda_neg=self.lambda_neg,
#         )


def triplet_w2v_loss(anchor: Tensor, positives: Tensor, negatives: Tensor,
                     lambda_pos=1.0, lambda_neg=1.0, reduction='mean'):
    n_dim = anchor.dim()
    assert n_dim == 2, f'expecting 2 dimensions for anchor (tensor size = {anchor.size()})'  # B x vec_dim

    if positives.dim() == n_dim:
        positives = positives.unsqueeze(1)
    if negatives.dim() == n_dim:
        negatives = negatives.unsqueeze(1)

    n_batch = anchor.size(0)
    vec_dim = anchor.size(1)
    anchor = anchor.unsqueeze(1)  # := matrix transposed (1st dim is batch)

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

    return loss_pos + loss_neg

# class TripletW2VLoss(TripletLoss):
#     def forward(self, anchor: Tensor, positives: Tensor, negatives: Tensor):
#         return triplet_w2v_loss(
#             anchor=anchor, positives=positives, negatives=negatives,
#             margin=self.margin, lambda_pos=self.lambda_pos, lambda_neg=self.lambda_neg,
#         )
