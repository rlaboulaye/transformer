import argparse

from torch.optim import Adam

from meta.stacked_optimizer import StackedOptimizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--config_path', type=str, default='schema/train_optimizer_config.json')
    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        verbose_print(verbose, vars(args))

    config_path = args.config_path
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    validate_against_schema(config, schema_path='schema/train_optimizer_config_schema.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize language model

    task_head = None
    optimizer = StackedOptimizer(task_head)
    optimizer.to(device)
    meta_optimizer = Adam(optimizer.parameters(), lr=args.meta_lr)