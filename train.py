import os
import math
import json
import argparse

import numpy as np
from scipy.stats import mode
import torch
from torch import nn
from torch.optim import Adam, SGD

from utils import set_seed, get_device, validate_against_schema, get_iterator, verbose_print
from logger import Logger
from data.text_encoder import TextEncoder
from data.data_utils import get_dataloaders
from model.double_head_model import DoubleHeadModel
from opt import OpenAIAdam
from loss import compute_double_head_loss, compute_accuracy


def score(dataloader, model, lm_criterion, task_criterion, lm_coef, task_coef, verbose):
	losses = []
	accuracies = []
	with torch.no_grad():
		model.eval()
		for x, m, y in get_iterator(dataloader, verbose):
			lm_logits, task_logits = model(x)
			double_head_loss, task_loss, lm_loss = compute_double_head_loss(x, y, m, lm_logits, task_logits, lm_criterion, task_criterion, lm_coef, task_coef)
			accuracy = compute_accuracy(y, task_logits)
			losses.extend([double_head_loss.cpu().item()] * x.shape[0])
			accuracies.extend([accuracy.cpu().item()] * x.shape[0])
	return np.mean(losses), np.mean(accuracies)

def run_epoch(train_dataloader, validation_dataloader, model, lm_criterion, task_criterion, lm_coef, task_coef, optimizer, scores_per_epoch, verbose):
	train_losses = []
	train_accuracies = []
	validation_losses = []
	validation_accuracies = []

	model.train()
	n_updates = 0
	for x, m, y in get_iterator(train_dataloader, verbose):
		lm_logits, task_logits = model(x)
		double_head_loss, task_loss, lm_loss = compute_double_head_loss(x, y, m, lm_logits, task_logits, lm_criterion, task_criterion, lm_coef, task_coef)
		double_head_loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		n_updates += 1

		if n_updates % math.ceil(float(len(train_dataloader)) / float(scores_per_epoch)) == 0 or n_updates == len(train_dataloader):
			train_loss, train_accuracy = score(train_dataloader, model, lm_criterion, task_criterion, lm_coef, task_coef, verbose=verbose)
			validation_loss, validation_accuracy = score(validation_dataloader, model, lm_criterion, task_criterion, lm_coef, task_coef, verbose=verbose)
			train_losses.append(train_loss)
			train_accuracies.append(train_accuracy)
			validation_losses.append(validation_loss)
			validation_accuracies.append(validation_accuracy)

	return train_losses, train_accuracies, validation_losses, validation_accuracies

def train(train_dataloader, validation_dataloader, model, lm_criterion, task_criterion, model_opt, logger, config):
	
	min_loss = float('inf')
	weight_directory = os.path.join('weights', logger.task_name)
	if not os.path.exists(weight_directory):
		os.makedirs(weight_directory)
	transformer_path = os.path.join(weight_directory, 'transformer.pth')
	lm_head_path = os.path.join(weight_directory, 'lm_head.pth')
	task_head_path = os.path.join(weight_directory, 'task_head.pth')

	for epoch in range(config['n_iter']):

		verbose_print(verbose, 'Running epoch {}'.format(epoch))
		
		train_losses, train_accuracies, validation_losses, validation_accuracies = run_epoch(train_dataloader, validation_dataloader, model, lm_criterion, task_criterion, config['lm_coef'], 1., model_opt, logger.results['scores_per_epoch'], verbose)
		logger.results['train_losses'].extend(train_losses)
		logger.results['train_accuracies'].extend(train_accuracies)
		logger.results['validation_losses'].extend(validation_losses)
		logger.results['validation_accuracies'].extend(validation_accuracies)

		logger.log()
		logger.plot()

		verbose_print(verbose, 'Train Loss: {}'.format(train_losses))
		verbose_print(verbose, 'Train Accuracy: {}'.format(train_accuracies))
		verbose_print(verbose, 'Validation Loss: {}'.format(validation_losses))
		verbose_print(verbose, 'Validation Accuracy: {}'.format(validation_accuracies))

		new_loss = np.mean(validation_losses)
		if new_loss < min_loss:
			min_loss = np.mean(validation_losses)
			torch.save(model.transformer.state_dict(), transformer_path)
			torch.save(model.lm_head.state_dict(), lm_head_path)
			torch.save(model.task_head.state_dict(), task_head_path)

	if min_loss != new_loss:
		model.transformer.load_state_dict(torch.load(transformer_path))
		model.lm_head.load_state_dict(torch.load(lm_head_path))
		model.task_head.load_state_dict(torch.load(task_head_path))

def test(test_dataloader, model, lm_criterion, task_criterion, logger, config):

	verbose_print(verbose, 'Testing')

	test_loss, test_accuracy = score(test_dataloader, model, lm_criterion, task_criterion, config['lm_coef'], 1., verbose)
	logger.results['test_loss'] = test_loss
	logger.results['test_accuracy'] = test_accuracy
	logger.log()

	verbose_print(verbose, 'Test Loss: {}'.format(test_loss))
	verbose_print(verbose, 'Test Accuracy: {}'.format(test_accuracy))

def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12, n_embd=768, path='./model_params/',
		path_names='./'):
	import re
	# Load weights from TF model
	verbose_print(verbose, "Loading weights...")
	names = json.load(open(path_names + 'parameters_names.json'))
	shapes = json.load(open(path + 'params_shapes.json'))
	offsets = np.cumsum([np.prod(shape) for shape in shapes])
	init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
	init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
	init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
	if n_ctx > 0:
		init_params[0] = init_params[0][:n_ctx]
	if n_special > 0:
		init_params[0] = np.concatenate(
			[init_params[1],
			(np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
			init_params[0]
			], 0)
	else:
		init_params[0] = np.concatenate(
			[init_params[1],
			init_params[0]
			], 0)
	del init_params[1]
	if n_transfer == -1:
		n_transfer = 0
	else:
		n_transfer = 1 + n_transfer * 12
	init_params = [arr.squeeze() for arr in init_params]

	try:
		assert model.embed.weight.shape == init_params[0].shape
	except AssertionError as e:
		e.args += (model.embed.weight.shape, init_params[0].shape)
		raise

	model.embed.weight.data = torch.from_numpy(init_params[0])

	for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
		name = name[6:]  # skip "model/"
		assert name[-2:] == ":0"
		name = name[:-2]
		name = name.split('/')
		pointer = model
		for m_name in name:
			if re.fullmatch(r'[A-Za-z]+\d+', m_name):
				l = re.split(r'(\d+)', m_name)
			else:
				l = [m_name]
			pointer = getattr(pointer, l[0])
			if len(l) >= 2:
				num = int(l[1])
				pointer = pointer[num]
		try:
			assert pointer.shape == ip.shape
		except AssertionError as e:
			e.args += (pointer.shape, ip.shape)
			raise
		pointer.data = torch.from_numpy(ip)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--config_path', type=str, default='schema/train_config.json')

	args = parser.parse_args()

	verbose = args.verbose
	if verbose:
		verbose_print(verbose, vars(args))

	config_path = args.config_path
	with open(config_path, 'r') as config_file:
		config = json.load(config_file)
	validate_against_schema(config, schema_path='schema/train_config_schema.json')

	task_path = config['task_path']
	with open(task_path, 'r') as task_file:
		task = json.load(task_file)
	validate_against_schema(task, schema_path='schema/task_schema.json')
	task_type = task['task_type']

	set_seed(config['seed'])
	device = get_device(verbose)

	text_encoder = TextEncoder(config['encoder_path'], config['bpe_path'])

	train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(task, text_encoder, config['test_split'], config['validation_split'], config['batch_size'], device, verbose)

	sequence_dim = train_dataloader.dataset.sequence_dim
	vocab_size = len(text_encoder.encoder) + sequence_dim

	dh_model = DoubleHeadModel(config, text_encoder.classify_token, task, vocab_size, sequence_dim)

	#
	load_openai_pretrained_model(dh_model.transformer, n_ctx=sequence_dim, n_special=3)
	# torch.save(dh_model.state_dict(), 'weights.pth')
	# dh_model = DoubleHeadModel(config, text_encoder.classify_token, task_type, vocab_size, sequence_dim)
	# verbose_print('Loading Weights')
	# dh_model.load_state_dict(torch.load('weights.pth'))
	#

	#

	lm_criterion = nn.CrossEntropyLoss(reduction='none')

	if task_type == 'MultipleChoice' or task_type == 'DocumentClassification':
		task_criterion = nn.CrossEntropyLoss(reduction='none')
	elif task_type == 'DocumentSimilarity':
		raise NotImplementedError()
	else:
		raise NotImplementedError()

	if config['opt'] == 'adam':
		model_opt = Adam(dh_model.parameters(),
						lr=config['lr'],
						betas=(config['b1'], config['b2']),
						eps=config['eps'])
	elif config['opt'] == 'openai_adam':
		n_updates_total = (train_dataloader.dataset.instances.shape[0] // config['batch_size']) * config['n_iter']
		model_opt = OpenAIAdam(dh_model.parameters(),
							   lr=config['lr'],
							   schedule=config['lr_schedule'],
							   warmup=config['lr_warmup'],
							   t_total=n_updates_total,
							   b1=config['b1'],
							   b2=config['b2'],
							   e=config['eps'],
							   l2=config['l2'],
							   vector_l2=config['vector_l2'],
							   max_grad_norm=config['max_grad_norm'])
	elif config['opt'] == 'sgd':
		model_opt = SGD(dh_model.parameters(),
						lr=config['lr'])
	else:
		raise NotImplementedError()

	dh_model.to(device)

	task_file_name = os.path.basename(config['task_path'])
	task_name = os.path.join(os.path.splitext(task_file_name)[0],
							'{}tr__{}val__{}te'.format(train_dataloader.dataset.instances.shape[0],
												validation_dataloader.dataset.instances.shape[0],
												test_dataloader.dataset.instances.shape[0])
		)
	targets = np.concatenate([train_dataloader.dataset.targets, validation_dataloader.dataset.targets, test_dataloader.dataset.targets])
	default_accuracy = float(mode(targets).count[0]) / float(len(targets))
	scores_per_epoch = config['scores_per_epoch']
	logger = Logger(config, task_name, scores_per_epoch, default_accuracy)

	train(train_dataloader, validation_dataloader, dh_model, lm_criterion, task_criterion, model_opt, logger, config)
	test(test_dataloader, dh_model, lm_criterion, task_criterion, logger, config)

	#TODO: calculate sequence_dim from both train and test
	#TODO: add number of classes to schema for document classification
