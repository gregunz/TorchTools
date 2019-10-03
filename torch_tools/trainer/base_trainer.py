from abc import abstractmethod
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from test_tube import Experiment, HyperOptArgumentParser

from .. import utils
from ..lightning_module import BaseModule


class BaseTrainer:
    def __init__(self, strategy: str = 'grid_search'):
        self.parser = self._init_parser(strategy)
        self.argz = self.parser.parse_args()
        self._set_seed()
        self.modules = OrderedDict(self._init_modules(self.argz))
        self.trainers = TrainerList()

    @abstractmethod
    def _init_modules(self, argz) -> List[Tuple[str, BaseModule]]:
        raise NotImplementedError

    @abstractmethod
    def _add_argz(self, parser: HyperOptArgumentParser):
        raise NotImplementedError

    def _set_seed(self) -> None:
        """
        It set the seed to most random number generators given by argz.manual_seed.
        If argz.manual_seed is None, it uses the clock to set a seed and overwrite
        the value of argz.manual_seed.

        :return:
        """
        argz = vars(self.argz)
        if argz.get('manual_seed', None) is not None:
            print('Setting the seed manually: {}'.format(argz['manual_seed']))
        else:
            argz['manual_seed'] = datetime.now().microsecond
            print('Setting a random seed from the clock: {}'.format(argz['manual_seed']))

        seed = argz['manual_seed']
        utils.set_seed(seed)

    def _init_parser(self, strategy: str):
        parser = HyperOptArgumentParser(strategy=strategy)

        default_epochs = 100
        parser.add_argument('--epochs', type=int, default=default_epochs,
                            help=f'number of epochs to train (default: {default_epochs})')

        default_log_interval = 100
        parser.add_argument('--log_interval', type=int, default=default_log_interval,
                            help=f'number of batches aggregated before logging the loss (default: {default_log_interval})')

        default_log_dir = Path('/data/logs/')
        parser.add_argument("--log_dir", type=str, default=default_log_dir,
                            help=f'directory for log outputs (tensorboard and more) (default: {default_log_dir})')

        default_seed = None  # this means it will be set by the clock (random)
        parser.add_argument('--manual_seed', type=int, default=default_seed,
                            help=f'set the seed manually for more reproducibility (default: {default_seed})')

        default_gpus = 1  # represents which gpu is used in binary representation (5 = 0101 = gpu0 and gpu2)
        parser.add_argument('--gpus', type=int, default=default_gpus,
                            help=f'which cuda device is used in binary representation '
                                 f'(i.e. 5 = 0101 = cuda:0 and cuda:2) (default: {default_gpus})')

        default_version = None  # when None, it creates a new one
        parser.add_argument('--version', type=int, default=default_version,
                            help=f'specify version continue its training (default: {default_version})')

        default_model_dir = Path('/data/models/')
        parser.add_argument('--model_dir', type=str, default=default_model_dir,
                            help=f'directory for model weights (default: {default_model_dir})')

        self._add_argz(parser)
        return parser

    def fit(self, module, exp_name, version=None, epochs=None):
        if version is None:
            version = self.argz.version
        if epochs is None:
            epochs = self.argz.epochs


        exp = Experiment(
            save_dir=self.argz.log_dir,
            name=exp_name,
            version=version,
        )
        exp.argparse(argparser=self.argz)
        module.cpu().add_graph(exp)

        ckpt_path = Path(self.argz.model_dir) / exp_name / f'version_{exp.version}'
        checkpoint_callback = ModelCheckpoint(
            filepath=ckpt_path,
            prefix=f'weights',
            verbose=True,
            #save_best_only=False,
            #monitor='val_loss',
            #mode='min',
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='auto'
        )

        trainer = pl.Trainer(
            experiment=exp,
            max_nb_epochs=epochs,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            gpus=utils.int_to_flags(self.argz.gpus),
            nb_sanity_val_steps=0,
        )
        self.trainers[exp_name, exp.version] = trainer
        trainer.fit(module)
        return exp.version

    def test(self, module, exp_name, version=None):
        self.trainers[exp_name, version].test(module)

    def default_run(self):
        for exp_name, module in self.modules.items():
            self.fit(module, exp_name)
            if module.has_tst_data():
                print('TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING ')
                self.test(module, exp_name)


class TrainerList:
    def __init__(self):
        self.trainers = dict()

    def __getitem__(self, expname_version):
        exp_name, version = self._decouple(expname_version)
        return self.trainers[exp_name][version]

    def __setitem__(self, expname_version, trainer):
        exp_name, version = self._decouple(expname_version)
        exp_trainers = self.trainers.get(exp_name, dict())
        self.trainers[exp_name] = exp_trainers
        exp_trainers[version] = trainer

    def _latest_version(self, exp_name):
        exp_trainer = self.trainers[exp_name]
        return max(exp_trainer.keys())

    def _decouple(self, expname_version):
        exp_name, version = expname_version
        assert isinstance(exp_name, str)
        if version is None:
            version = self._latest_version(exp_name)
        else:
            version = int(version)
        assert 0 <= version
        return exp_name, version
