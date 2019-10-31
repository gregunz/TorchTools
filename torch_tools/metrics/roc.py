import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics


def compute_roc_auc(outputs, targets):
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    outputs = outputs.squeeze()
    targets = targets.squeeze()
    assert outputs.dim() == 1
    assert targets.dim() == 1
    assert outputs.size() == targets.size()
    target_values = targets.unique()
    assert target_values.size(0) == 2
    assert set(target_values.tolist()) == {0, 1}

    targets = targets.detach().cpu()
    outputs = outputs.detach().cpu()

    fpr, tpr, threshold = metrics.roc_curve(targets, outputs)
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, threshold, auc


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
