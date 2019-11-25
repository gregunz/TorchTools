"""
This modules contains the main buiding blocks for training a PyTorch models.

- Models are trained following a `Strategy`
- Training is handled by the `Executor`

Typically, one should only worry about designing the `Strategy` as the
`Executor` is simply here to facilitate and optimize the training loop.

Example:
Design a classification `Strategy` by defining
- Training Data: create a dataloader
- Training Step: feed input to your model and compute the loss
- [Optional] Validation step: feed input to your model and compute the loss and accuracy, log them to tensorboard
- [Optional] Testing step
...

"""
from torch_tools.training.callback import Callback
from torch_tools.training.executor import Executor
from torch_tools.training.strategy import Strategy
