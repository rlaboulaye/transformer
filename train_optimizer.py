import os
import json
import argparse

from torch.optim import Adam

from meta.stacked_optimizer import StackedOptimizer
from utils import set_seed, get_device, validate_against_schema, get_iterator, verbose_print


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--config_path', type=str, default='schema/train_optimizer_config.json')
    parser.add_argument('--task_directory_path', type=str)
    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        verbose_print(verbose, vars(args))

    config_path = args.config_path
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    validate_against_schema(config, schema_path='schema/train_optimizer_config_schema.json')

    set_seed(config['seed'])
    device = get_device(verbose)

    # initialize language model

    # task_head = None
    # optimizer = StackedOptimizer(task_head)
    # optimizer.to(device)
    # meta_optimizer = Adam(optimizer.parameters(), lr=config['meta_lr'])

    tasks = os.listdir(args.task_directory_path)

    for meta_epoch in range(config['meta_epochs']):
        verbose_print(verbose, 'Running meta-epoch {}'.format(meta_epoch))