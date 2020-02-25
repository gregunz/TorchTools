from argparse import ArgumentParser
from pathlib import Path

import invertransforms as T
from classification_strategy import ClassifierStrategy
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

from torch_tools.data.datasets.util import split
from torch_tools.data.samplers import RandomSampler
from torch_tools.models.vision import SimpLeNet
from torch_tools.training.executors import SimpleExecutor as Executor


def main(**kwargs):
    ########################
    # [DATA] some datasets #
    ########################

    dataset_name = kwargs['dataset_name']

    tf = T.Compose([
        T.ToTensor(),
        T.TransformIf(T.Normalize(mean=[0.13], std=[0.31]), dataset_name == 'mnist'),
        T.TransformIf(T.Normalize(mean=[0.5] * 3, std=[0.5] * 3), dataset_name == 'cifar10'),
    ])

    dataset, tst_dataset = {
        'mnist': lambda: (
            MNIST(root='/data/', train=True, transform=tf, download=True),
            MNIST(root='/data/', train=False, transform=tf, download=True),
        ),
        'cifar10': lambda: (
            CIFAR10(root='/data/', train=True, transform=tf, download=True),
            CIFAR10(root='/data/', train=False, transform=tf, download=True),
        )}[dataset_name]()

    tng_dataset, val_dataset = split(dataset, percentage=kwargs['val_percentage'])
    sampler = RandomSampler(tng_dataset, num_samples=len(val_dataset), replacement=False)
    tng_dataloader = DataLoader(dataset=tng_dataset, batch_size=kwargs['tng_batch_size'], sampler=sampler,
                                num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=kwargs['val_batch_size'], shuffle=False, num_workers=4)
    tst_dataloader = DataLoader(dataset=tst_dataset, batch_size=kwargs['val_batch_size'], shuffle=False, num_workers=4)

    ###########################
    # [MODEL] a pytorch model #
    ###########################

    sample_input, _ = dataset[0]
    net = SimpLeNet(
        input_size=sample_input.size(),
        n_classes=10,
    )

    ########################################
    # [STRATEGY] it describes the training #
    ########################################

    exp_name = f'{net.__class__.__name__.lower()}/{dataset_name}'
    kwargs['log_dir'] = Path(kwargs['log_dir']) / exp_name

    classifier = ClassifierStrategy(net=net, **kwargs)

    ##################################
    # [EXECUTOR] it handles the rest #
    ##################################

    executor = Executor(
        tng_dataloader=tng_dataloader,
        val_dataloader=val_dataloader,
        tst_dataloader=tst_dataloader,
        exp_name=exp_name,
        **kwargs,
    )

    executor.train_test(strategy=classifier, **kwargs)


if __name__ == '__main__':
    ########################
    # [ARGS] some defaults #
    ########################

    parser = ArgumentParser()
    ClassifierStrategy.add_argz(parser)
    Executor.add_argz(parser)
    parser.add_argument('--val_percentage', type=float, default=0.1)
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--tng_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    argz = parser.parse_args()
    main(**vars(argz))
