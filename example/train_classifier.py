import invertransforms as T
from classification_strategy import ClassifierStrategy
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

from torch_tools.datasets.util import split
from torch_tools.models.vision import SimpLeNet
from torch_tools.training.executors import LightningExecutor

if __name__ == '__main__':
    parser = HyperOptArgumentParser()

    ########################
    # [ARGS] some defaults #
    ########################

    ClassifierStrategy.add_argz(parser)
    LightningExecutor.add_argz(parser)
    parser.add_argument('--val_percentage', type=float, default=0.1)
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--tng_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    argz = parser.parse_args()

    ########################
    # [DATA] some datasets #
    ########################

    tf = T.Compose([
        T.ToTensor(),
        T.TransformIf(T.Normalize(mean=[0.13], std=[0.31]), argz.dataset_name == 'mnist'),
        T.TransformIf(T.Normalize(mean=[0.5] * 3, std=[0.5] * 3), argz.dataset_name == 'cifar10'),
    ])

    dataset, tst_dataset = {
        'mnist': lambda: (
            MNIST(root='/data/', train=True, transform=tf, download=True),
            MNIST(root='/data/', train=False, transform=tf, download=True),
        ),
        'cifar10': lambda: (
            CIFAR10(root='/data/', train=True, transform=tf, download=True),
            CIFAR10(root='/data/', train=False, transform=tf, download=True),
        )}[argz.dataset_name]()

    tng_dataset, val_dataset = split(dataset, percentage=argz.val_percentage)
    tng_dataloader = DataLoader(dataset=tng_dataset, batch_size=argz.tng_batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=argz.val_batch_size, shuffle=False, num_workers=4)
    tst_dataloader = DataLoader(dataset=tst_dataset, batch_size=argz.val_batch_size, shuffle=False, num_workers=4)

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

    classifier = ClassifierStrategy(
        tng_dataloader=tng_dataloader,
        val_dataloader=val_dataloader,
        tst_dataloader=tst_dataloader,
        net=net,
        **vars(argz),
    )

    ##################################
    # [EXECUTOR] it handles the rest #
    ##################################

    exp_name = f'{net.__class__.__name__.lower()}/{argz.dataset_name}'
    executor = LightningExecutor(exp_name=exp_name, **vars(argz))

    executor.train(strategy=classifier, **vars(argz))
    executor.test()
