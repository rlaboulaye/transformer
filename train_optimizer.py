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
from meta_logger import MetaLogger


def no_grad(module):
    for parameter in module.parameters():
        parameter.requires_grad = False

def freeze_weights(model, num_layers):
    no_grad(model.transformer.embed)
    for layer in model.transformer.h[:num_layers]:
        no_grad(layer)

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
    train_evaluator = Evaluator(lm_criterion, task_criterion, config['lm_coef'], 1., target_type)
    test_evaluator = Evaluator(lm_criterion, task_criterion, 0., 1., target_type)

    dh_model = DoubleHeadModel(config, text_encoder.classify_token, num_output, vocab_size, sequence_dim)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=sequence_dim, n_special=3, verbose=verbose)
    dh_model.to(device)

    return dh_model, (train_dataloader, test_dataloader), (train_evaluator, test_evaluator)

def meta_train_instance(optimizer, task, module_index, config, meta_config, text_encoder, device, verbose):
    dh_model, dataloaders, evaluators = prepare_experiment(config, task, text_encoder, device, verbose)
    train_dataloader, test_dataloader = dataloaders
    train_evaluator, test_evaluator = evaluators
    freeze_weights(dh_model, num_layers=meta_config['num_frozen_layers'])

    optimizer.initialize_params(dh_model, learn_initialization_indices)
    optimizer.reset_state()

    for epoch in range(config['n_iter']):
        tuned_dh_model = train_epoch(dh_model, optimizer, train_dataloader, train_evaluator, module_index, verbose)
    loss, accuracy = test_epoch(tuned_dh_model, test_dataloader, test_evaluator, verbose)

    meta_optimizer.zero_grad()
    loss.backward()
    meta_optimizer.step()

    return loss.cpu().item(), accuracy

def meta_test_instance(optimizer, task, config, meta_config, text_encoder, device, verbose):
    set_seed(config['seed'])
    dh_model, dataloaders, evaluators = prepare_experiment(config, task, text_encoder, device, verbose)
    train_dataloader, test_dataloader = dataloaders
    train_evaluator, test_evaluator = evaluators
    freeze_weights(dh_model, num_layers=meta_config['num_frozen_layers'])

    optimizer.initialize_params(dh_model, learn_initialization_indices)
    optimizer.reset_state()

    for epoch in range(config['n_iter']):
        tuned_dh_model = train_epoch(dh_model, optimizer, train_dataloader, train_evaluator, None, verbose)
    loss, accuracy = test_epoch(tuned_dh_model, test_dataloader, test_evaluator, verbose)

    return loss.cpu().item(), accuracy

def meta_test_instance_alternative_optimizer(optimizer_class, optimizer_arguments, task, config, meta_config, text_encoder, device, verbose):
    set_seed(config['seed'])
    dh_model, dataloaders, evaluators = prepare_experiment(config, task, text_encoder, device, verbose)
    train_dataloader, test_dataloader = dataloaders
    train_evaluator, test_evaluator = evaluators
    freeze_weights(dh_model, num_layers=meta_config['num_frozen_layers'])

    optimizer = optimizer_class(dh_model.parameters(), **optimizer_arguments)

    for epoch in range(config['n_iter']):
        tuned_dh_model = train_epoch(dh_model, optimizer, train_dataloader, train_evaluator, None, verbose)
    loss, accuracy = test_epoch(tuned_dh_model, test_dataloader, test_evaluator, verbose)

    return loss.cpu().item(), accuracy

def meta_test_instance_baseline(task, config, text_encoder, device, verbose):
    set_seed(config['seed'])
    dh_model, dataloaders, evaluators = prepare_experiment(config, task, text_encoder, device, verbose)
    train_dataloader, test_dataloader = dataloaders
    train_evaluator, test_evaluator = evaluators
    freeze_weights(dh_model, num_layers=12)

    optimizer = Adam(dh_model.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']), eps=config['eps'])

    for epoch in range(config['n_iter']):
        tuned_dh_model = train_epoch(dh_model, optimizer, train_dataloader, train_evaluator, None, verbose)
    loss, accuracy = test_epoch(tuned_dh_model, test_dataloader, test_evaluator, verbose)

    return loss.cpu().item(), accuracy

def train_epoch(dh_model, optimizer, dataloader, evaluator, module_index, verbose):
    for x, m, y in get_iterator(dataloader, verbose):
        lm_logits, task_logits = dh_model(x)
        double_head_loss, task_loss, lm_loss = evaluator.compute_double_head_loss(x, y, m, lm_logits, task_logits)
        dh_model.zero_grad()
        double_head_loss.backward()
        if isinstance(optimizer, StackedOptimizer):
            tuned_dh_model = optimizer(dh_model, double_head_loss, module_index)
        else:
            optimizer.step()
            tuned_dh_model = dh_model
    return tuned_dh_model

def test_epoch(dh_model, dataloader, evaluator, verbose):
    losses = []
    accuracies = []
    for x, m, y in get_iterator(dataloader, verbose):
        lm_logits, task_logits = dh_model(x)
        double_head_loss, task_loss, lm_loss = evaluator.compute_double_head_loss(x, y, m, lm_logits, task_logits)
        accuracy = test_evaluator.compute_score(y, task_logits)
        losses.append(double_head_loss)
        accuracies.append(accuracy.cpu().item())
    losses = torch.cat([loss.unsqueeze(-1) for loss in losses], dim=-1)
    loss = losses.mean(-1)
    accuracy = np.mean(accuracies)
    return loss, accuracy

def meta_train_epoch(logger, optimizer, tasks, config, meta_config, text_encoder, device, verbose):
    verbose_print(verbose, 'Training')
    accuracies = []
    losses = []
    for path in tasks:
        verbose_print(verbose, 'Task {}'.format(path))
        task = get_document(os.path.join(args.task_directory_path, path), 'schema/task_schema.json')
        for module_index in range(len(optimizer.optimizers)):
            verbose_print(verbose, 'Module index {}'.format(module_index))
            loss, accuracy = meta_train_instance(optimizer, task, module_index, config, meta_config, text_encoder, device, verbose)
            accuracies.append(accuracy)
            losses.append(loss)
            verbose_print(verbose, 'Base Epoch Test Accuracy: {}'.format(accuracy))
            verbose_print(verbose, 'Base Epoch Test Loss: {}'.format(loss))
    train_accuracy = np.mean(accuracies)
    train_loss = np.mean(losses)
    logger.results['train_accuracies'].append(train_accuracy)
    logger.results['train_losses'].append(train_loss)
    verbose_print(verbose, 'Meta Epoch Train Accuracy: {}'.format(train_accuracy))
    verbose_print(verbose, 'Meta Epoch Train Loss: {}'.format(train_loss))

def meta_validation_epoch(logger, optimizer, tasks, config, meta_config, text_encoder, device, verbose):
    verbose_print(verbose, 'Validation')
    accuracies = []
    losses = []
    for path in tasks:
        verbose_print(verbose, 'Task {}'.format(path))
        task = get_document(os.path.join(args.task_directory_path, path), 'schema/task_schema.json')
        loss, accuracy = meta_test_instance(optimizer, task, config, meta_config, text_encoder, device, verbose)
        accuracies.append(accuracy)
        losses.append(loss)
        verbose_print(verbose, 'Base Epoch Test Accuracy: {}'.format(accuracy))
        verbose_print(verbose, 'Base Epoch Test Loss: {}'.format(loss))
    validation_accuracy = np.mean(accuracies)
    validation_loss = np.mean(losses)
    logger.results['validation_accuracies'].append(validation_accuracy)
    logger.results['validation_losses'].append(validation_loss)
    verbose_print(verbose, 'Meta Epoch Validation Accuracy: {}'.format(validation_accuracy))
    verbose_print(verbose, 'Meta Epoch Validation Loss: {}'.format(validation_loss))

def meta_test_epoch(logger, optimizer, tasks, config, meta_config, text_encoder, device, verbose):
    verbose_print(verbose, 'Testing')
    baseline_accuracies = []
    sgd_accuracies = []
    adam_accuracies = []
    stacked_optimizer_accuracies = []
    baseline_losses = []
    sgd_losses = []
    adam_losses = []
    stacked_optimizer_losses = []
    for path in tasks:
        verbose_print(verbose, 'Task {}'.format(path))
        task = get_document(os.path.join(args.task_directory_path, path), 'schema/task_schema.json')
        baseline_loss, baseline_accuracy = meta_test_instance_baseline(task, config, text_encoder, device, verbose)
        sgd_loss, sgd_accuracy = meta_test_instance_alternative_optimizer(SGD, {'lr':config['lr']}, task, config, meta_config, text_encoder, device, verbose)
        adam_loss, adam_accuracy = meta_test_instance_alternative_optimizer(Adam, {'lr':config['lr'], 'betas':(config['b1'], config['b2']), 'eps':config['eps']}, task, config, meta_config, text_encoder, device, verbose)
        stacked_optimizer_loss, stacked_optimizer_accuracy = meta_test_instance(optimizer, task, config, meta_config, text_encoder, device, verbose)
        baseline_accuracies.append(baseline_accuracy)
        sgd_accuracies.append(sgd_accuracy)
        adam_accuracies.append(adam_accuracy)
        stacked_optimizer_accuracies.append(stacked_optimizer_accuracy)
        baseline_losses.append(baseline_loss)
        sgd_losses.append(sgd_loss)
        adam_losses.append(adam_loss)
        stacked_optimizer_losses.append(stacked_optimizer_loss)
        verbose_print(verbose, 'Base Epoch Test Accuracies: {} (Baseline), {} (SGD), {} (Adam), {} (StackedOptimizer)'.format(baseline_accuracy, sgd_accuracy, adam_accuracy, stacked_optimizer_accuracy))
        verbose_print(verbose, 'Base Epoch Test Loss: {} (Baseline), {} (SGD), {} (Adam), {} (StackedOptimizer)'.format(baseline_loss, sgd_loss, adam_loss, stacked_optimizer_loss))
    logger.results['baseline_test_loss'] = np.mean(baseline_losses)
    logger.results['baseline_test_accuracy'] = np.mean(baseline_accuracies)
    logger.results['sgd_test_loss'] = np.mean(sgd_losses)
    logger.results['sgd_test_accuracy'] = np.mean(sgd_accuracies)
    logger.results['adam_test_loss'] = np.mean(adam_losses)
    logger.results['adam_test_accuracy'] = np.mean(adam_accuracies)
    logger.results['stacked_optimizer_test_loss'] = np.mean(stacked_optimizer_losses)
    logger.results['stacked_optimizer_test_accuracy'] = np.mean(stacked_optimizer_accuracies)
    verbose_print(verbose, 'Meta Test Accuracies: {} (Baseline), {} (SGD), {} (Adam), {} (StackedOptimizer)'.format(np.mean(baseline_accuracies), np.mean(sgd_accuracies), np.mean(adam_accuracies), np.mean(stacked_optimizer_accuracies)))
    verbose_print(verbose, 'Meta Test Loss: {} (Baseline), {} (SGD), {} (Adam), {} (StackedOptimizer)'.format(np.mean(baseline_losses), np.mean(sgd_losses), np.mean(adam_losses), np.mean(stacked_optimizer_losses)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--config_path', type=str, default='schema/train_optimizer_config.json')
    parser.add_argument('--task_directory_path', type=str)
    args = parser.parse_args()

    log = args.log
    verbose = args.verbose
    if verbose:
        verbose_print(verbose, vars(args))

    meta_config = get_document(args.config_path, 'schema/train_optimizer_config_schema.json')
    config = get_document(meta_config['train_config_path'], 'schema/train_config_schema.json')

    set_seed(meta_config['seed'])
    device = get_device(verbose)

    text_encoder = TextEncoder(config['encoder_path'], config['bpe_path'])
    tasks = os.listdir(args.task_directory_path)
    meta_config['meta_test_split']
    meta_config['meta_validation_split']
    test_tasks_size = round(len(tasks) * meta_config['meta_test_split'])
    validation_tasks_size = round((len(tasks) - test_tasks_size) * meta_config['meta_validation_split'])
    test_tasks, validation_tasks, train_tasks = np.split(tasks, [test_tasks_size, test_tasks_size + validation_tasks_size])

    task_path = args.task_directory_path + np.random.choice(train_tasks)
    task = get_document(task_path, 'schema/task_schema.json')
    dh_model, dataloaders, evaluators = prepare_experiment(config, task, text_encoder, device, verbose)
    train_dataloader, test_dataloader = dataloaders
    train_evaluator, test_evaluator = evaluators
    freeze_weights(dh_model, num_layers=meta_config['num_frozen_layers'])

    learn_initialization_indices = []
    optimizer = StackedOptimizer(dh_model, learn_initialization_indices=learn_initialization_indices)
    optimizer.to(device)
    meta_optimizer = Adam(optimizer.parameters(), lr=meta_config['meta_lr'])

    logger = MetaLogger(meta_config, args.task_directory_path)

    for meta_epoch in range(meta_config['meta_epochs']):
        verbose_print(verbose, 'Running meta-epoch {}'.format(meta_epoch))
        meta_train_epoch(logger, optimizer, train_tasks, config, meta_config, text_encoder, device, verbose)
        meta_validation_epoch(logger, optimizer, validation_tasks, config, meta_config, text_encoder, device, verbose)
        if log:
            logger.log()
            logger.plot()
            torch.save(optimizer.state_dict(), os.path.join(logger.results_directory, 'weights_{}.pth'.format(meta_epoch)))
    meta_test_epoch(logger, optimizer, test_tasks, config, meta_config, text_encoder, device, verbose)
    if log:
        logger.log()
        logger.plot()
