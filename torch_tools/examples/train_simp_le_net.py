from typing import List, Tuple

from anode import utils
from anode.datasets.clf.sup import MNIST, CIFAR10, SupervisedDataset
from anode.models import SimpLeNet
from anode.modules.classification_module import ClassificationModule

from torch_tools.datasets.util import split
from torch_tools.lightning_module import BaseModule
from torch_tools.trainer.base_trainer import BaseTrainer


class SimpLeNetTrainer(BaseTrainer):
    def _add_argz(self, parser):
        SupervisedDataset.add_args(parser)
        ClassificationModule.add_args(parser)
        default_val_percentage = 0.1
        parser.add_argument('--val_percentage', type=float, default=default_val_percentage,
                            help=f'percentage of data used for validation (default: {default_val_percentage})')

    def _init_modules(self, argz) -> List[Tuple[str, BaseModule]]:
        dataset, tst_dataset = {
            'mnist': lambda: (MNIST(train=True), MNIST(train=False)),
            'cifar10': lambda: (CIFAR10(train=True), CIFAR10(train=False)),
        }[argz.dataset_name]()

        print('TRAINING DATASET:')
        dataset.print_stats()
        print('TESTING DATASET:')
        tst_dataset.print_stats()

        simp_le_net = SimpLeNet(input_size=dataset.size(), n_classes=dataset.n_classes)

        clf_name = f'{utils.unsupervised_name}/{argz.dataset_name}/{simp_le_net.__class__.__name__}'.lower()

        tng_dataset, val_dataset = split(dataset, percentage=argz.val_percentage)
        classifier = ClassificationModule(
            tng_dataset=tng_dataset,
            val_dataset=val_dataset,
            tst_dataset=tst_dataset,
            classifier=simp_le_net,
            lr=argz.lr,
            betas=argz.betas,
            tng_batch_size=argz.tng_batch_size,
            val_batch_size=argz.val_batch_size,
        )

        return [
            (clf_name, classifier)
        ]


if __name__ == '__main__':
    simple_trainer = SimpLeNetTrainer()
    simple_trainer.default_run()
