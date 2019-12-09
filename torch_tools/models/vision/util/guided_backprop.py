import copy
from collections import Sequence

import torch
from torch import nn


def backprop_relu(module, grad_in, grad_out):
    if isinstance(module, nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.0),)


class GuidedBackProp:
    def __init__(self, model, device=None):
        self.device = device
        if self.device is None:
            self.device = next(model.parameters()).device

        self.model = copy.deepcopy(model).eval().to(self.device)
        for _, module in self.model.named_modules():
            module.register_backward_hook(backprop_relu)

    def __call__(self, x, class_idx=None, norm=True):
        in_device = x.device
        x = x.clone().to(self.device).requires_grad_()
        class_idx = self.class_backward(self.model, x, class_idx)
        bp_cam = x.grad

        if norm:
            bp_cam = self.positive_norm(bp_cam)

        return bp_cam.to(in_device), class_idx.view(-1).to(in_device)

    @staticmethod
    def positive_norm(tensor: torch.Tensor):
        b = tensor.size(0)
        tensor = tensor.clamp(min=0)
        tensor -= tensor.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        tensor /= tensor.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        return tensor

    @staticmethod
    def class_backward(model, x, class_idx):
        output = model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1, keepdim=True)
        elif isinstance(class_idx, int):
            class_idx = torch.tensor(class_idx).view(1, 1).expand(x.size(0), 1).to(x.device)
        elif isinstance(class_idx, Sequence):
            class_idx = torch.tensor(class_idx).view(-1, 1)

        one_hot = torch.zeros_like(output, requires_grad=True).scatter(1, class_idx, 1)
        one_hot = torch.sum(one_hot * output)
        one_hot.backward()
        return class_idx
