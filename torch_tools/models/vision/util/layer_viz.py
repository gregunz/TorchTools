import copy
from collections import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def positive_norm(tensor: torch.Tensor):
    b = tensor.size(0)
    tensor = tensor.clamp(min=0)
    tensor -= tensor.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
    tensor /= tensor.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
    return tensor


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
        class_idx = class_backward(self.model, x, class_idx)
        bp_cam = x.grad

        if norm:
            bp_cam = positive_norm(bp_cam)

        return bp_cam.to(in_device), class_idx.view(-1).to(in_device)


class GradCAM:
    def __init__(self, model, target_layer_name, device=None):
        self.device = device
        if self.device is None:
            self.device = next(model.parameters()).device

        self.model = copy.deepcopy(model).eval().to(self.device)
        self.model.zero_grad()
        self.feature_map = None
        self.grad = None

        if target_layer_name not in {name for name, _ in self.model.named_modules()}:
            raise ValueError('No such target layer name.')

        for name, module in self.model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self.save_feature_map)
                module.register_backward_hook(self.save_grad)

    def save_feature_map(self, module, input, output):
        self.feature_map = output.detach()

    def save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def __call__(self, x, class_idx=None, norm=True):
        in_device = x.device
        w, h = x.size()[-2:]
        x = x.clone().requires_grad_().to(self.device)
        self.model.zero_grad()
        class_idx = class_backward(self.model, x, class_idx)

        weights = self.grad.mean(dim=(2, 3))
        print(weights.size())
        print(self.feature_map.size())
        weights = self.feature_map * weights.view(weights.size() + (1, 1))
        weights = weights.sum(dim=1).clamp(min=0)
        weights = F.interpolate(weights.unsqueeze(1), size=(w, h), mode='bilinear', align_corners=True)

        if norm:
            weights = positive_norm(weights)

        return weights.to(in_device), class_idx.view(-1).to(in_device)
