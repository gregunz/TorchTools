from typing import List

import torch


class AggFn:
    def __init__(self, outputs: List[dict]):
        self.outputs = outputs

    def stack(self, metric_name):
        return torch.stack([x[metric_name] for x in self.outputs], dim=0)

    def cat(self, metric_name):
        return torch.cat([x[metric_name] for x in self.outputs], dim=0)
