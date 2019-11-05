from pathlib import Path
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from torch_tools.training.strategies import SimpleStrategy
from torch_tools.training.util import ImageLogger


class AEStrategy(SimpleStrategy, ImageLogger):
    def __init__(self, auto_encoder: nn.Module, lr: float, betas: Tuple[float, float], log_dir: Union[str, Path],
                 output_to_img=None, **kwargs):
        SimpleStrategy.__init__(
            self,
            net=auto_encoder,
            lr=lr, betas=betas,
            log_dir=log_dir,
        )
        ImageLogger.__init__(self, output_to_image=output_to_img)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def tng_step(self, batch, batch_idx, optimizer_idx, epoch_idx, num_batches: int) -> dict:
        # forward pass
        x, _ = batch  # ignoring label
        x_hat = self.net(x)

        loss = self.loss(x_hat, x)

        self.log(
            metrics_dict={'training/loss': loss, },
            global_step=num_batches * epoch_idx + batch_idx,
            interval=20,
        )

        return {
            'loss': loss,
        }

    def val_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int, num_batches: int) -> dict:
        # forward pass
        x, _ = batch  # ignoring label
        x_hat = self.net(x)
        losses = torch.mean(((x - x_hat) ** 2).view(x.size(0), -1), dim=1)

        self.log(
            metrics_dict={
                'validation/batch/loss': losses.mean(),
                'validation/batch/loss_std': losses.std(),
            },
            global_step=num_batches * epoch_idx + batch_idx,
            interval=10,
        )

        if self.output_to_img is not None and batch_idx == 0:
            self.log_images(
                tag='validation/in_out',
                images_tensor=torch.stack((x, x_hat)),
                global_step=epoch_idx,
            )

        return {
            'losses': losses,
        }

    def val_agg_outputs(self, outputs, agg_fn, epoch_idx) -> dict:
        losses = agg_fn.cat('losses')
        loss = losses.mean()
        logs = {
            'validation/epoch/loss': loss,
            'validation/epoch/loss_std': losses.std(),
        }
        self.log(logs, global_step=epoch_idx)
        return {
            'val_loss': loss.item()
        }
