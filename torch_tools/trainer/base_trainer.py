from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
from test_tube import Experiment, HyperOptArgumentParser

from torch_tools import utils
from ..lightning_module import BaseModule

AddFn = Callable[[HyperOptArgumentParser], None]


class BaseTrainer(ABC):
    def __init__(self, strategy: str, add_dataset_fn: AddFn, add_module_fn: AddFn):
        parser = self.init_parser(strategy)

        add_dataset_fn(parser)
        add_module_fn(parser)

        self.parser = parser

    @abstractmethod
    def init_module(self, argz) -> BaseModule:
        raise NotImplementedError

    def init_parser(self, strategy: str) -> HyperOptArgumentParser:
        parser = HyperOptArgumentParser(strategy=strategy)

        default_epochs = 50
        parser.add_argument('--epochs', type=int, default=default_epochs,
                            help=f'number of epochs to train (default: {default_epochs})')

        default_log_interval = 100
        parser.add_argument('--log_interval', type=int, default=default_log_interval,
                            help=f'number of batches aggregated before logging the loss (default: {default_log_interval})')

        default_log_dir = Path('/data/logs/')
        parser.add_argument("--log_dir", type=str, default=default_log_dir,
                            help=f'log directory for log outputs (tensorboard and more) (default: {default_log_dir})')

        default_seed = None  # this means it will be set by the clock (random)
        parser.add_argument('--manual_seed', type=int, default=default_seed,
                            help=f'set the seed manually for more reproducibility (default: {default_seed})')

        default_gpus = 1  # represents which gpu is used in binary representation (5 = 0101 = gpu0 and gpu2)
        parser.add_argument('--gpus', type=int, default=default_gpus,
                            help=f'which cuda device is used in binary representation '
                                 f'(i.e. 5 = 0101 = cuda:0 and cuda:2) (default: {default_gpus})')
        return parser

    def start(self, do_test=True):
        argz = self.parser.parse_args()
        utils.set_seed_from_argz(argz)
        module = self.init_module(argz)

        exp = Experiment(
            save_dir=argz.log_dir,
            name=module.name.lower(),
        )
        exp.argparse(argparser=argz)
        # exp.tag(tag_dict=module.infos())
        module.add_graph(exp)

        trainer = pl.Trainer(
            experiment=exp,
            max_nb_epochs=argz.epochs,
            gpus=utils.int_to_flags(argz.gpus),
            nb_sanity_val_steps=0,
            # distributed_backend='dp',
        )

        trainer.fit(module)

        if module.has_tst_data():
            trainer.test(module)

        return module, trainer
