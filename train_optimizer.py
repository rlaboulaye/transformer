import os
import json
import argparse

import numpy as np
from torch.optim import Adam

from meta.stacked_optimizer import StackedOptimizer
from utils import set_seed, get_device, validate_against_schema, get_iterator, verbose_print

#
import math

import numpy as np
from scipy.stats import mode
import torch
from torch import nn
from torch.optim import Adam, SGD

from logger import Logger
from data.text_encoder import TextEncoder
from data.data_utils import get_dataloaders
from model.double_head_model import DoubleHeadModel
from opt import OpenAIAdam
from loss import compute_double_head_loss, compute_accuracy
from train import load_openai_pretrained_model
#


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--config_path', type=str, default='schema/train_optimizer_config.json')
    parser.add_argument('--task_directory_path', type=str)
    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        verbose_print(verbose, vars(args))

    meta_config_path = args.config_path
    with open(meta_config_path, 'r') as meta_config_file:
        meta_config = json.load(meta_config_file)
    validate_against_schema(meta_config, schema_path='schema/train_optimizer_config_schema.json')

    config_path = meta_config['train_config_path']
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    validate_against_schema(config, schema_path='schema/train_config_schema.json')

    set_seed(meta_config['seed'])
    device = get_device(verbose)

    # initialize language model

    # task_head = None
    # optimizer = StackedOptimizer(task_head)
    # optimizer.to(device)
    # meta_optimizer = Adam(optimizer.parameters(), lr=meta_config['meta_lr'])

    #
    text_encoder = TextEncoder(config['encoder_path'], config['bpe_path'])

    # load_openai_pretrained_model(dh_model.transformer, n_ctx=sequence_dim, n_special=3)
    #

    tasks = os.listdir(args.task_directory_path)

    for meta_epoch in range(meta_config['meta_epochs']):
        verbose_print(verbose, 'Running meta-epoch {}'.format(meta_epoch))
        task_path = args.task_directory_path + np.random.choice(tasks)

        #
        with open(task_path, 'r') as task_file:
            task = json.load(task_file)
        validate_against_schema(task, schema_path='schema/task_schema.json')
        task_type = task['task_type']
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(task, text_encoder, config['test_split'], config['validation_split'], config['batch_size'], device, verbose)

        for batch in train_dataloader:
            continue
        for batch in test_dataloader:
            continue
        #