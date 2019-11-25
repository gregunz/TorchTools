from collections import OrderedDict

from torch import nn


def smart_load_state_dict(model: nn.Module, state_dict: dict) -> None:
    # change the keys names to "force" them to match
    state_dict = OrderedDict(zip(model.state_dict().keys(), state_dict.values()))
    model.load_state_dict(state_dict)
