import random
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_seed_from_argz(argz: Namespace, verbosity: int = 1, arg_name: str = 'manual_seed') -> None:
    """
    It set the seed to most random number generators given by argz.manual_seed.
    If argz.manual_seed is None, it uses the clock to set a seed and overwrite
    the value of argz.manual_seed.

    :param argz: args from argparser.parse()
    :param verbosity:
    :param arg_name:
    :return:
    """
    argz = vars(argz)
    if argz.get(arg_name, None) is not None and verbosity >= 1:
        print('Setting the seed manually: {}'.format(argz.manual_seed))
    else:
        argz[arg_name] = datetime.now().microsecond
        if verbosity >= 2:
            print('Setting a random seed from the clock: {}'.format(argz.manual_seed))

    seed = argz[arg_name]
    set_seed(seed)