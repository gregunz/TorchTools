import torch


def gen_pr_curves(predictions: torch.Tensor, labels: torch.Tensor):
    n_classes = predictions.size(1)
    for c in range(n_classes):
        class_prob = predictions[:, c]
        rng = list(set(range(n_classes)) - {c})
        max_prob = predictions[:, rng].max(dim=1)[0]
        bin_preds = class_prob / (class_prob + max_prob)
        # torch.softmax(torch.stack([class_prob, max_prob], dim=1), dim=1)[:, 0]
        yield bin_preds, (labels == c).long()
