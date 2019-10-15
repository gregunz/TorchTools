import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from torch_tools.train import SimpleStrategy


class ClassifierStrategy(SimpleStrategy):
    """
    A strategy to train classification models.

    It applies a LogSoftmax on the logits and optimizes the Negative Log Likelihood Loss.
    It also logs useful metrics.
    """

    def forward(self, x):
        logits = self.net(x)
        return F.log_softmax(logits, dim=1)

    def loss(self, output, target):
        return F.nll_loss(output, target)

    def tng_step(self, batch, batch_idx, optimizer_idx, epoch_idx):
        # forward pass
        x, y = batch
        y_hat = self.forward(x)
        return self.loss(y_hat, y)

    def evaluate_step(self, data_batch):
        x, y = data_batch
        y_hat = self.forward(x)
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

    def val_step(self, batch, batch_idx, optimizer_idx, epoch_idx):
        outputs = self.evaluate_step(data_batch=batch)
        # logging outputs
        val_step = batch_idx + epoch_idx * len(self.val_data_loader())
        outputs_to_log = {
            'validation/batch/loss': outputs['loss'].item(),
            'validation/batch/accuracy': outputs['acc'].item(),
        }
        self.log(outputs_to_log, global_step=val_step)
        return outputs

    def val_agg_outputs(self, outputs, agg_fn, epoch_idx) -> None:
        loss = agg_fn.stack('loss').mean()
        logs = {
            'validation/epoch/loss': loss,
            'validation/epoch/accuracy': agg_fn.stack('acc').mean(),
        }
        self.log(logs, global_step=epoch_idx)

        # pr_curves = metrics.gen_pr_curves(agg_fn.cat('pred'), agg_fn.cat('gt'))
        # for c, (preds, labels) in enumerate(pr_curves):
        #     self.logger.add_pr_curve(
        #         tag=f'validation/pr_curve/{c}',
        #         predictions=preds,
        #         labels=labels,
        #         global_step=epoch_idx,
        #     )
        # return loss

    def tst_step(self, batch, batch_idx, optimizer_idx) -> dict:
        return self.evaluate_step(data_batch=batch)

    def tst_agg_outputs(self, outputs, agg_fn):
        tst_acc = agg_fn.stack('acc').mean()
        self.logger.add_text(
            tag='test/accuracy',
            text_string=f'{tst_acc:.4f}'
        )

        # pr_curves = list(metrics.gen_pr_curves(agg_fn.cat('pred'), agg_fn.cat('gt')))
        # # fix: because tensorboard do not let change the step if test has only one step and val has multiple
        # for i in range(self.current_epoch + 1):
        #     for c, (preds, labels) in enumerate(pr_curves):
        #         self.experiment.add_pr_curve(
        #             tag=f'test/pr_curve/{c}',
        #             predictions=preds,
        #             labels=labels,
        #             global_step=i,
        #         )
