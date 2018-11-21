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


def freeze_weights(model, num_layers):
    for parameter in model.transformer.embed.parameters():
        parameter.requires_grad = False
    for layer in model.transformer.h[:num_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    ###
    # for parameter in model.transformer.h[-1].attn.parameters():
    #     parameter.requires_grad = False
    # for parameter in model.transformer.h[-1].mlp.c_fc.parameters():
    #     parameter.requires_grad = False
    # for parameter in model.transformer.h[-1].mlp.c_proj.parameters():
    #     parameter.requires_grad = False
    # for parameter in model.transformer.h[-1].ln_1.parameters():
    #     parameter.requires_grad = False
    # for parameter in model.transformer.h[-1].ln_2.parameters():
    #     parameter.requires_grad = False
    # for parameter in model.task_head.parameters():
    #     parameter.requires_grad = False

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

    text_encoder = TextEncoder(config['encoder_path'], config['bpe_path'])
    tasks = os.listdir(args.task_directory_path)

    #
    task_path = args.task_directory_path + np.random.choice(tasks)
    with open(task_path, 'r') as task_file:
        task = json.load(task_file)
    validate_against_schema(task, schema_path='schema/task_schema.json')
    task_type = task['task_type']
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(task, text_encoder, config['test_split'], config['validation_split'], config['batch_size'], device, verbose)
    sequence_dim = train_dataloader.dataset.sequence_dim
    vocab_size = len(text_encoder.encoder) + sequence_dim
    
    dh_model = DoubleHeadModel(config, text_encoder.classify_token, task, vocab_size, sequence_dim)
    freeze_weights(dh_model, num_layers=11)
    dh_model.to(device)

    modules = [module for module in dh_model.modules() if len([param for param in module._parameters.values() if param is not None and param.requires_grad]) > 0]
    learn_initialization_indices = []

    optimizer = StackedOptimizer(dh_model, learn_initialization_indices=learn_initialization_indices)
    optimizer.to(device)
    meta_optimizer = Adam(optimizer.parameters(), lr=.001)

    lm_criterion = nn.CrossEntropyLoss(reduction='none')
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    #

    test_losses = []
    for meta_epoch in range(meta_config['meta_epochs']):
        # verbose_print(verbose, 'Running meta-epoch {}'.format(meta_epoch))
        print('Running meta-epoch {}'.format(meta_epoch))
        for module_index in range(len(optimizer.optimizers)):
            # verbose_print(verbose, 'Module index {}'.format(module_index))
            print('Module index {}'.format(module_index))
            task_path = args.task_directory_path + np.random.choice(tasks)

            with open(task_path, 'r') as task_file:
                task = json.load(task_file)
            validate_against_schema(task, schema_path='schema/task_schema.json')
            task_type = task['task_type']
            train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(task, text_encoder, config['test_split'], config['validation_split'], config['batch_size'], device, verbose)
            sequence_dim = train_dataloader.dataset.sequence_dim
            vocab_size = len(text_encoder.encoder) + sequence_dim
            
            dh_model = DoubleHeadModel(config, text_encoder.classify_token, task, vocab_size, sequence_dim)
            load_openai_pretrained_model(dh_model.transformer, n_ctx=sequence_dim, n_special=3, verbose=verbose)
            freeze_weights(dh_model, num_layers=11)
            dh_model.to(device)

            optimizer.initialize_params(dh_model, learn_initialization_indices)
            optimizer.reset_state()

            for x, m, y in get_iterator(train_dataloader, verbose):
                lm_logits, task_logits = dh_model(x)
                double_head_loss, task_loss, lm_loss = compute_double_head_loss(x, y, m, lm_logits, task_logits, lm_criterion, task_criterion, config['lm_coef'], 1.)
                dh_model.zero_grad()
                double_head_loss.backward()
                tuned_dh_model = optimizer(dh_model, double_head_loss, module_index)
            losses = []
            for x, m, y in get_iterator(validation_dataloader, verbose):
                lm_logits, task_logits = tuned_dh_model(x)
                double_head_loss, task_loss, lm_loss = compute_double_head_loss(x, y, m, lm_logits, task_logits, lm_criterion, task_criterion, config['lm_coef'], 1.)
                losses.append(double_head_loss)
            losses = torch.cat([loss.unsqueeze(-1) for loss in losses], dim=-1)
            loss = losses.mean(-1)
            meta_optimizer.zero_grad()
            loss.backward()
            meta_optimizer.step()
            test_loss = loss.cpu().item()
            test_losses.append(test_loss)
            print('Epoch Test Loss: {}'.format(test_loss))
            print('Mean Test Loss (last 20): {}'.format(np.mean(test_losses[-20:])))