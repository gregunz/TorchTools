import torch
from torch.nn import functional as F

from torch_tools.training.strategies import SimpleStrategy


class ClassifierStrategy(SimpleStrategy):
    """
    A strategy to training classification models.

    It applies a LogSoftmax on the logits and optimizes the Negative Log Likelihood Loss.
    It also logs useful metrics.
    """
    def loss(self, output, target):
        return F.nll_loss(F.log_softmax(output, dim=1), target)

    def tng_step(self, batch, batch_idx, optimizer_idx, epoch_idx, num_batches: int) -> dict:
        # forward pass
        x, y = batch
        logits = self.net(x)
        labels_hat = torch.argmax(logits, dim=1)

        loss = self.loss(logits, y)
        acc = torch.sum(y == labels_hat).float() / y.size(0)
        return {
            'loss': loss,
            'acc': acc
        }

    def evaluate_step(self, data_batch):
        x, y = data_batch
        logits = self.net(x)
        loss_val = self.loss(logits, y)

        # acc
        labels_hat = torch.argmax(logits, dim=1)
        val_acc = torch.sum(y == labels_hat).float() / y.size(0)
        return {
            # 'gt': y,
            # 'logits': logits,
            'loss': loss_val,
            'acc': val_acc,
        }

    def val_step(self, batch, batch_idx, epoch_idx, num_batches) -> dict:
        outputs = self.evaluate_step(data_batch=batch)
        return outputs

    def val_agg_outputs(self, outputs, agg_fn, epoch_idx) -> dict:
        loss = agg_fn.stack('loss').mean()
        acc = agg_fn.stack('acc').mean()
        logs = {
            'validation/epoch/loss': loss,
            'validation/epoch/accuracy': acc,
        }
        self.log(logs, global_step=epoch_idx)
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def tst_step(self, batch, batch_idx, num_batches) -> dict:
        return self.evaluate_step(data_batch=batch)

    def tst_agg_outputs(self, outputs, agg_fn) -> dict:
        tst_acc = agg_fn.stack('acc').mean()
        self.logger.add_text(
            tag='test/accuracy',
            text_string=f'{tst_acc:.4f}'
        )
        return {
            'tst_acc': tst_acc
        }
