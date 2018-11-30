import os
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam

from meta.stacked_optimizer import StackedOptimizer
from utils import set_seed, get_device, validate_against_schema, get_iterator, verbose_print
from data.text_encoder import TextEncoder
from data.data_utils import get_dataloaders
from model.double_head_model import DoubleHeadModel
from evaluate import Evaluator
from train import load_openai_pretrained_model


def freeze_weights(model, num_layers):
    for parameter in model.transformer.embed.parameters():
        parameter.requires_grad = False
    for layer in model.transformer.h[:num_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

def get_document(config_path, schema_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    validate_against_schema(config, schema_path=schema_path)
    return config

def prepare_experiment(config, task, text_encoder, device, verbose):
    train_dataloader, validation_dataloader, test_dataloader, document_structure = get_dataloaders(task, text_encoder, config['test_split'], config['validation_split'], config['batch_size'], device, verbose, sequence_dim=config['sequence_dim'])
    max_position_encoding = train_dataloader.dataset.max_position_encoding
    sequence_dim = train_dataloader.dataset.sequence_dim
    vocab_size = len(text_encoder.encoder) + max_position_encoding
    num_output = task['target']['num_classes'] if not document_structure == 'one_to_many' else 1
    
    target_type = task['target']['target_type']
    if target_type == 'classification':
        task_criterion = nn.CrossEntropyLoss(reduction='none')
    elif target_type == 'regression':
        task_criterion = nn.MSELoss(reduction='none')
    lm_criterion = nn.CrossEntropyLoss(reduction='none')
    evaluator = Evaluator(lm_criterion, task_criterion, config['lm_coef'], 1., target_type)

    dh_model = DoubleHeadModel(config, text_encoder.classify_token, num_output, vocab_size, sequence_dim)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=sequence_dim, n_special=3, verbose=verbose)
    dh_model.to(device)

    return dh_model, (train_dataloader, validation_dataloader, test_dataloader), evaluator


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--config_path', type=str, default='schema/train_optimizer_config.json')
    parser.add_argument('--task_directory_path', type=str)
    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        verbose_print(verbose, vars(args))

    meta_config = get_document(args.config_path, 'schema/train_optimizer_config_schema.json')
    config = get_document(meta_config['train_config_path'], 'schema/train_config_schema.json')

    set_seed(meta_config['seed'])
    device = get_device(verbose)

    text_encoder = TextEncoder(config['encoder_path'], config['bpe_path'])
    tasks = os.listdir(args.task_directory_path)

    task_path = args.task_directory_path + np.random.choice(tasks)
    task = get_document(task_path, 'schema/task_schema.json')
    dh_model, dataloaders, evaluator = prepare_experiment(config, task, text_encoder, device, verbose)
    train_dataloader, validation_dataloader, test_dataloader = dataloaders
    freeze_weights(dh_model, num_layers=11)

    learn_initialization_indices = []
    optimizer = StackedOptimizer(dh_model, learn_initialization_indices=learn_initialization_indices)
    optimizer.to(device)
    meta_optimizer = Adam(optimizer.parameters(), lr=meta_config['meta_lr'])

    test_losses = []
    for meta_epoch in range(meta_config['meta_epochs']):
        # verbose_print(verbose, 'Running meta-epoch {}'.format(meta_epoch))
        print('Running meta-epoch {}'.format(meta_epoch))
        for module_index in range(len(optimizer.optimizers)):
            # verbose_print(verbose, 'Module index {}'.format(module_index))
            print('Module index {}'.format(module_index))

            task_path = args.task_directory_path + np.random.choice(tasks)
            print(os.path.basename(task_path))
            task = get_document(task_path, 'schema/task_schema.json')
            dh_model, dataloaders, evaluator = prepare_experiment(config, task, text_encoder, device, verbose)
            train_dataloader, validation_dataloader, test_dataloader = dataloaders
            freeze_weights(dh_model, num_layers=11)

            optimizer.initialize_params(dh_model, learn_initialization_indices)
            optimizer.reset_state()

            for x, m, y in get_iterator(train_dataloader, verbose):
                lm_logits, task_logits = dh_model(x)
                double_head_loss, task_loss, lm_loss = evaluator.compute_double_head_loss(x, y, m, lm_logits, task_logits)
                dh_model.zero_grad()
                double_head_loss.backward()
                tuned_dh_model = optimizer(dh_model, double_head_loss, module_index)
            losses = []
            accuracies = []
            for x, m, y in get_iterator(validation_dataloader, verbose):
                lm_logits, task_logits = tuned_dh_model(x)
                double_head_loss, task_loss, lm_loss = evaluator.compute_double_head_loss(x, y, m, lm_logits, task_logits)
                accuracy = evaluator.compute_score(y, task_logits)
                losses.append(double_head_loss)
                accuracies.append(accuracy.cpu().item())
            losses = torch.cat([loss.unsqueeze(-1) for loss in losses], dim=-1)
            loss = losses.mean(-1)
            meta_optimizer.zero_grad()
            loss.backward()
            meta_optimizer.step()
            test_loss = loss.cpu().item()
            test_losses.append(test_loss)
            test_accuracy = np.mean(accuracies)
            print('Epoch Test Accuracy: {}'.format(test_accuracy))
            print('Epoch Test Loss: {}'.format(test_loss))
            print('Mean Test Loss (last 20): {}'.format(np.mean(test_losses[-20:])))