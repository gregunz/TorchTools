import torch
from anode import metrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from torch_tools.trainer.lightning_module import SimpleModule


class ClassificationModule(SimpleModule):
    def __init__(self,
                 tng_dataset: Dataset,
                 val_dataset: Dataset,
                 tst_dataset: Dataset,
                 classifier: nn.Module,
                 lr, betas, tng_batch_size, val_batch_size):
        super().__init__(
            tng_dataset=tng_dataset,
            val_dataset=val_dataset,
            tst_dataset=tst_dataset,
            net=classifier,

            lr=lr,
            betas=betas,
            tng_batch_size=tng_batch_size,
            val_batch_size=val_batch_size,
        )

    def forward(self, x):
        logits = self.net(x)
        return F.log_softmax(logits, dim=1)

    def loss(self, output, target):
        return F.nll_loss(output, target)

    def tng_step(self, batch, batch_idx, optimizer_idx):
        # forward pass
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def evaluate_step(self, data_batch):
        x, y = data_batch
        y_hat = self(x)
        loss_val = self.loss(y_hat, y)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).float() / y.size(0)
        return {
            'gt': y,
            'pred': y_hat.exp(),
            'loss': loss_val,
            'acc': val_acc,
        }

    def val_step(self, batch, batch_idx, optimizer_idx, global_step):
        outputs = self.evaluate_step(data_batch=batch)
        # logging outputs
        val_step = batch_idx + self.current_epoch * len(self.val_dataloader)
        outputs_to_log = {
            'validation/batch/loss': outputs['loss'].item(),
            'validation/batch/accuracy': outputs['acc'].item(),
        }
        self.experiment.log(outputs_to_log, global_step=val_step)
        return outputs

    def val_agg_outputs(self, outputs, agg_fn) -> torch.Tensor:
        loss = agg_fn.stack('loss').mean()
        logs = {
            'validation/epoch/loss': loss,
            'validation/epoch/accuracy': agg_fn.stack('acc').mean(),
        }
        self.experiment.log(logs, global_step=self.current_epoch)

        pr_curves = metrics.gen_pr_curves(agg_fn.cat('pred'), agg_fn.cat('gt'))
        for c, (preds, labels) in enumerate(pr_curves):
            self.experiment.add_pr_curve(
                tag=f'validation/pr_curve/{c}',
                predictions=preds,
                labels=labels,
                global_step=self.current_epoch,
            )
        return loss

    def tst_step(self, batch, batch_idx, optimizer_idx):
        return self.evaluate_step(data_batch=batch)

    def tst_agg_outputs(self, outputs, agg_fn):
        tst_acc = agg_fn.stack('acc').mean()
        self.experiment.add_text(
            tag='test/accuracy',
            text_string=f'{tst_acc:.4f}'
        )

        pr_curves = list(metrics.gen_pr_curves(agg_fn.cat('pred'), agg_fn.cat('gt')))
        # fix: because tensorboard do not let change the step if test has only one step and val has multiple
        for i in range(self.current_epoch + 1):
            for c, (preds, labels) in enumerate(pr_curves):
                self.experiment.add_pr_curve(
                    tag=f'test/pr_curve/{c}',
                    predictions=preds,
                    labels=labels,
                    global_step=i,
                )
