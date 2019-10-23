import torch
from anode.strategies import AEStrategy
from torch.nn import functional as F


class VAEStrategy(AEStrategy):
    kl_coef = 1  # todo: smarter strategies are possible

    def loss(self, output, target):
        x_hat, mu, logvar = output
        mse_loss = F.mse_loss(x_hat, target)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse_loss, kld_loss / x_hat.nelement()

    def tng_step(self, batch, batch_idx, optimizer_idx, epoch_idx) -> dict:
        # forward pass
        x, _ = batch  # ignoring label
        output = self.net(x)
        mse_loss, kld_loss = self.loss(output, x)
        loss = mse_loss + kld_loss * self.kl_coef

        self.log({
            'training/batch/loss': loss,
            'training/batch/loss_mse': mse_loss,
            'training/batch/loss_kld': kld_loss,
        }, global_step=self.num_tng_batch * epoch_idx + batch_idx)

        return {
            'loss': loss,
        }

    def val_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        x, _ = batch  # ignoring label
        output = self.net(x)
        mse_loss, kld_loss = self.loss(output, x)
        val_loss = mse_loss + kld_loss * self.kl_coef

        self.log({
            'validation/batch/loss': val_loss,
            'validation/batch/loss_mse': mse_loss,
            'validation/batch/loss_kld': kld_loss,
        }, global_step=self.num_val_batch * epoch_idx + batch_idx)

        x_hat, _, _ = output  # logging output image
        if self.output_to_img is not None and batch_idx == 0:
            self.log_images(
                tag='validation/in_out',
                images_tensor=torch.stack((x, x_hat)),
                global_step=epoch_idx,
            )

        return {
            'loss': val_loss,
            'loss_mse': mse_loss,
            'loss_kld': kld_loss,
        }

    def val_agg_outputs(self, outputs, agg_fn, epoch_idx) -> dict:
        losses = agg_fn.stack('loss')
        loss = losses.mean()
        logs = {
            'validation/epoch/loss': loss,
            'validation/epoch/loss_std': losses.std(),
            'validation/epoch/loss_mse': agg_fn.stack('loss_mse').mean(),
            'validation/epoch/loss_kld': agg_fn.stack('loss_kld').mean(),
        }
        self.log(logs, global_step=epoch_idx)
        return {
            'val_loss': loss
        }
