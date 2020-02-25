from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def new_parser(**kwargs):
    args = {
        'conflict_handler': 'resolve',
        'formatter_class': ArgumentDefaultsHelpFormatter,
    }
    args.update(kwargs)
    return ArgumentParser(**args)
