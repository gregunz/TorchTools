from typing import Union, List, Tuple

from torch import nn


def cnn_out_size(h: int, w: int, cnn_model: nn.Module) -> (int, int):
    """

    :param h:
    :param w:
    :param cnn_model:
    :return:
    """
    hw = [h, w]
    cnn_modules = flatten_modules(cnn_model)
    for module in cnn_modules:
        if hasattr(module, 'kernel_size') and hasattr(module, 'stride') and hasattr(module, 'padding'):
            def to_pair(x: Union[int, Tuple[int]]) -> Tuple[int]:
                if isinstance(x, int):
                    x = (x, x)
                return x

            kernels = to_pair(module.kernel_size)
            strides = to_pair(module.stride)
            paddings = to_pair(module.padding)

            if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
                for i, k, s, p in zip(range(2), kernels, strides, paddings):
                    hw[i] = 1 + (hw[i] - k + 2 * p) // s
            elif isinstance(module, nn.ConvTranspose2d):
                for i, k, s, p in zip(range(2), kernels, strides, paddings):
                    hw[i] = s * (hw[i] - 1) + k - 2 * p
            else:
                raise NotImplementedError(module.__class__.__name__)
    h, w = hw
    return h, w


def flatten_modules(module: Union[nn.Module, list]) -> List[nn.Module]:
    s = module
    if isinstance(module, nn.Module):
        s = list(module._modules.values())
    else:
        assert isinstance(module, list)

    if len(s) == 0:
        return [module]
    elif len(s) == 1:
        return flatten_modules(s[0])
    else:
        return flatten_modules(s[0]) + flatten_modules(s[1:])
