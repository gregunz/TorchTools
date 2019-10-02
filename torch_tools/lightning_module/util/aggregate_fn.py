import torch


def stack_outputs(self, outputs):
    return lambda metric_name: torch.stack([x[metric_name] for x in outputs], dim=0)


def cat_outputs(self, outputs):
    return lambda metric_name: torch.cat([x[metric_name] for x in outputs], dim=0)