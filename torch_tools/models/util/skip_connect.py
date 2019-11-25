from typing import Tuple

import torch
from torch import nn


class SkipConnect(nn.Module):
    def __init__(self, blocks_down: Tuple[nn.Module], blocks_mid: Tuple[nn.Module], blocks_up: Tuple[nn.Module],
                 reduce):
        super().__init__()
        if len(blocks_down) != len(blocks_up):
            raise ValueError(
                f'Cannot skip-connect when num of downs ({len(blocks_down)}) != ({len(blocks_up)}) num of ups')
        self.blocks_down = nn.ModuleList(blocks_down)
        self.blocks_mid = nn.ModuleList(blocks_mid)
        self.blocks_up = nn.ModuleList(blocks_up)
        self.reduce = reduce

    @classmethod
    def from_blocks(cls, *blocks: nn.Module, reduce):
        n_blocks = len(blocks)
        down_end = n_blocks // 2
        up_start = n_blocks // 2 + 1
        if n_blocks % 2 == 0:  # if pair number of blocks then the middle block will made of sequential 2 blocks
            down_end = n_blocks // 2 - 1

        return cls(
            blocks_down=blocks[:down_end],
            blocks_mid=blocks[down_end:up_start],
            blocks_up=blocks[up_start:],
            reduce=reduce,
        )

    def forward(self, x) -> torch.Tensor:
        skips = []
        for block_down in self.blocks_down:
            x = block_down(x)
            skips = [x] + skips

        for block_mid in self.blocks_mid:
            x = block_mid(x)

        for block_up, skip_x in zip(self.blocks_up, skips):
            x = self.reduce(x, skip_x)
            x = block_up(x)

        return x
