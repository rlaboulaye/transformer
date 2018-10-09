import json
import argparse

import numpy as np
import torch
from torch import nn

from utils import set_seed, get_device, validate_task, get_iterator, verbose_print, log
from data.text_encoder import TextEncoder
from data.data_utils import get_dataloaders
from model.double_head_model import DoubleHeadModel
from opt import OpenAIAdam
from loss import compute_double_head_loss, compute_accuracy


def score(dataloader, model, loss_function):
	# Compute the accuracy
	logits = []
	cost = 0
	with torch.no_grad:
		model.eval()
		for x, m, y in get_iterator(dataloader):
			# TODO compute accuracy
			pass

def run_epoch(dataloader, model, lm_criterion, task_critetion, lm_coef, task_coef, optimizer=None, verbose=False):
	losses = []
	accuracies = []
	if optimizer is None:
		model.eval()
	else:
		model.train()
	for x, m, y in get_iterator(dataloader, verbose):
		lm_logits, task_logits = model(x)
		double_head_loss, task_loss, lm_loss = compute_double_head_loss(x, y, m, lm_logits, task_logits, lm_criterion, task_critetion, lm_coef, task_coef)
		if optimizer is not None:
			double_head_loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		accuracy = compute_accuracy(y, task_logits)
		losses.extend([double_head_loss.cpu().item()] * x.shape[0])
		accuracies.extend([accuracy.cpu().item()] * x.shape[0])
	return np.mean(losses), np.mean(accuracies)
	# Cloze expected output: 27.89827631632487, 21.38771795272827, 18.45592082977295
	# Airline expected output: 19.27845308886453

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
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--batch_size', type=int, default=8)
	# TODO Rename arguments
	parser.add_argument('--n_iter', type=int, default=3)
	parser.add_argument('--n_embd', type=int, default=768)
	parser.add_argument('--n_head', type=int, default=12)
	parser.add_argument('--n_layer', type=int, default=12)
	parser.add_argument('--embd_pdrop', type=float, default=.1)
	parser.add_argument('--attn_pdrop', type=float, default=.1)
	parser.add_argument('--resid_pdrop', type=float, default=.1)
	parser.add_argument('--clf_pdrop', type=float, default=.1)
	parser.add_argument('--afn', type=str, choices=['relu', 'swish', 'gelu'], default='gelu')
	#
	parser.add_argument('--lm_coef', type=float, default=0.5)
	parser.add_argument('--lr', type=float, default=6.25e-5)
	parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
	parser.add_argument('--lr_warmup', type=float, default=0.002)
	parser.add_argument('--b1', type=float, default=0.9)
	parser.add_argument('--b2', type=float, default=0.999)
	parser.add_argument('--e', type=float, default=1e-8)
	parser.add_argument('--l2', type=float, default=0.01)
	parser.add_argument('--vector_l2', action='store_true')
	parser.add_argument('--max_grad_norm', type=int, default=1)
	#
	parser.add_argument('--test_split', type=float, default=.2)
	parser.add_argument('--validation_split', type=float, default=.2)
	parser.add_argument('--encoder_path', type=str, default='model_params/encoder_bpe_40000.json')
	parser.add_argument('--bpe_path', type=str, default='model_params/vocab_40000.bpe')
	parser.add_argument('--task_path', type=str)

	args = parser.parse_args()
	verbose = args.verbose
	if verbose:
		verbose_print(verbose, args)

	task_path = args.task_path
	with open(task_path, 'r') as task_file:
		task = json.load(task_file)
	validate_task(task)
	task_type = task['task_type']

	set_seed(args.seed)
	device = get_device(verbose)

	text_encoder = TextEncoder(args.encoder_path, args.bpe_path)

	train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(task, text_encoder, args.test_split, args.validation_split, args.batch_size, device, verbose)

	sequence_dim = train_dataloader.dataset.sequence_dim
	vocab_size = len(text_encoder.encoder) + sequence_dim

	dh_model = DoubleHeadModel(args, text_encoder.classify_token, task_type, vocab_size, sequence_dim)

	#
	# load_openai_pretrained_model(dh_model.transformer, n_ctx=sequence_dim, n_special=3)
	# torch.save(dh_model.state_dict(), 'weights.pth')
	# dh_model = DoubleHeadModel(args, text_encoder.classify_token, task_type, vocab_size, sequence_dim)
	verbose_print('Loading Weights')
	dh_model.load_state_dict(torch.load('weights.pth'))
	#

	#
	criterion = nn.CrossEntropyLoss(reduction='none')
	n_updates_total = (train_dataloader.dataset.instances.shape[0] // args.batch_size) * args.n_iter
	model_opt = OpenAIAdam(dh_model.parameters(),
							lr=args.lr,
							schedule=args.lr_schedule,
							warmup=args.lr_warmup,
							t_total=n_updates_total,
							b1=args.b1,
							b2=args.b2,
							e=args.e,
							l2=args.l2,
							vector_l2=args.vector_l2,
							max_grad_norm=args.max_grad_norm)

	dh_model.to(device)

	# train model
	for epoch in range(args.n_iter):
		verbose_print(verbose, 'Running epoch {}'.format(epoch))
		verbose_print(verbose, 'Training')
		train_loss, train_accuracy = run_epoch(train_dataloader, dh_model, criterion, criterion, args.lm_coef, 1., optimizer=model_opt, verbose=verbose)
		verbose_print(verbose, 'Train Loss: {}'.format(train_loss))
		verbose_print(verbose, 'Train Accuracy: {}'.format(train_accuracy))
		verbose_print(verbose, 'Validation')
		with torch.no_grad():
			validation_loss, validation_accuracy = run_epoch(validation_dataloader, dh_model, criterion, criterion, args.lm_coef, 1., verbose=verbose)
		verbose_print(verbose, 'Validation Loss: {}'.format(validation_loss))
		verbose_print(verbose, 'Validation Accuracy: {}'.format(validation_accuracy))
	verbose_print(verbose, 'Testing')
	with torch.no_grad():
		test_loss, test_accuracy = run_epoch(test_dataloader, dh_model, criterion, criterion, args.lm_coef, 1., verbose=verbose)
	verbose_print(verbose, 'Test Loss: {}'.format(test_loss))
	verbose_print(verbose, 'Test Accuracy: {}'.format(test_accuracy))

	#TODO: calculate sequence_dim from both train and test
	#TODO: add number of classes to schema for document classification