import copy

from torch.nn import functional as F

from torch_tools.models.vision.util import GuidedBackProp


class GradCAM:
    def __init__(self, model, target_layer_name, device=None):
        self.device = device
        if self.device is None:
            self.device = next(model.parameters()).device

        self.model = copy.deepcopy(model).eval().to(self.device)
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
        class_idx = GuidedBackProp.class_backward(self.model, x, class_idx)

        weights = self.grad.mean(dim=(2, 3))
        print(weights.size())
        print(self.feature_map.size())
        weights = self.feature_map * weights.view(weights.size() + (1, 1))
        weights = weights.sum(dim=1).clamp(min=0)
        weights = F.interpolate(weights.unsqueeze(1), size=(w, h), mode='bilinear', align_corners=True)

        if norm:
            weights = GuidedBackProp.positive_norm(weights)

        return weights.to(in_device), class_idx.view(-1).to(in_device)
