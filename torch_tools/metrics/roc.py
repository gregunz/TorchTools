import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.nn.functional import pad


def _pad(tensor, value, start=0, end=0):
    return pad(tensor, pad=[start, end], value=value)


def _roc_asserts(outputs, targets, true_label):
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    outputs = outputs.squeeze()
    targets = targets.squeeze()
    assert outputs.dim() == 1
    assert targets.dim() == 1
    assert outputs.size() == targets.size(), f'{outputs.size()} != {targets.size()}'
    target_values = targets.unique()
    assert target_values.size(0) == 2, target_values
    assert set(target_values.tolist()) == {0, 1}, target_values
    assert true_label in set(target_values.tolist())


def compute_roc_auc(outputs, targets, true_label=1, thr_limit='+-1'):
    _roc_asserts(outputs, targets, true_label)

    _, output_sort_indices = outputs.sort()

    if true_label == 0:
        outputs = (-1) * outputs
        targets = 1 - targets
    else:
        output_sort_indices = output_sort_indices.flip([0])  # to be consistent with sklearn

    outputs_sorted = outputs[output_sort_indices]
    targets_sorted = targets[output_sort_indices]

    threshold_idxs = (targets_sorted[1:] - targets_sorted[:-1]).nonzero().view(-1)
    threshold_idxs = _pad(threshold_idxs, value=targets_sorted.shape[0] - 1, end=1)

    tps = targets_sorted.cumsum(0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    fpr = _pad(fps / fps[-1].to(outputs.dtype), value=0, start=1)
    tpr = _pad(tps / tps[-1].to(outputs.dtype), value=0, start=1)
    thresholds = 0.5 * (outputs_sorted[threshold_idxs[:-1]] + outputs_sorted[threshold_idxs[:-1] + 1])

    thresholds = _pad(thresholds, value=outputs_sorted[0] + 1, start=1, end=1)
    thresholds[-1] = outputs_sorted[-1] - 1

    if thr_limit is 'inf':
        thresholds[0] = float('+inf')
        thresholds[-1] = float('-inf')

    auc = compute_auc(x=fpr, y=tpr).item()

    if true_label == 0:
        fpr = fpr.flip([0])
        tpr = tpr.flip([0])
        thresholds = (-thresholds).flip([0])

    return fpr, tpr, thresholds, auc


def compute_roc_auc_numpy(outputs, targets, true_label=1):
    _roc_asserts(outputs, targets, true_label)

    targets = targets.detach().cpu()
    outputs = outputs.detach().cpu()

    if true_label == 0:
        outputs = (-1) * outputs
        targets = 1 - targets

    fpr, tpr, thresholds = metrics.roc_curve(targets, outputs)
    auc = np.trapz(tpr, fpr)

    fpr = torch.tensor(fpr)
    tpr = torch.tensor(tpr)
    thresholds = torch.tensor(thresholds)

    if true_label == 0:
        fpr = fpr.flip([0])
        tpr = tpr.flip([0])
        thresholds = (-thresholds).flip([0])

    return fpr, tpr, thresholds, auc


def compute_auc(x, y):
    dx = x - _pad(x, value=0, start=1)[:-1]
    return torch.sum(0.5 * (y[1:] + y[:-1]) * dx[1:])


def create_roc_figure(fpr, tpr):
    roc_figure = plt.figure(figsize=(5, 5))
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr)
    return roc_figure


def __deprecated_compute_roc_auc(outputs, targets, true_label=1):
    _roc_asserts(outputs, targets, true_label)

    outputs_sorted, output_sort_indices = outputs.sort()
    targets_sorted = targets[output_sort_indices]

    indices_altern = (targets_sorted[:-1] != targets_sorted[1:]).nonzero().squeeze()
    thresholds = 0.5 * (outputs_sorted[indices_altern] + outputs_sorted[indices_altern + 1])

    inf_min = torch.tensor([float('-inf')])
    inf_pos = torch.tensor([float('+inf')])
    thresholds = torch.cat([inf_min, thresholds, inf_pos]).flip([0])

    preds = (outputs_sorted.view(1, -1).expand(thresholds.size(0), -1) >= thresholds.view(-1, 1)).long()

    preds_true = (preds == targets_sorted).long()
    preds_false = 1 - preds_true

    pos_target_mask = targets_sorted == true_label
    neg_target_mask = ~pos_target_mask

    tpr = preds_true[:, pos_target_mask].float().mean(1)
    fpr = preds_false[:, neg_target_mask].float().mean(1)

    auc = compute_auc(x=fpr, y=tpr).item()

    return fpr, tpr, thresholds, auc
