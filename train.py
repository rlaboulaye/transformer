import json
import argparse

from text_encoder import TextEncoder
from utils import set_seed, get_device, validate_task
from data_utils import get_dataloaders
from model.double_head_model import DoubleHeadModel


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--verbose', type=bool, default=False)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--test_split', type=float, default=.2)
	parser.add_argument('--validation_split', type=float, default=.2)
	parser.add_argument('--encoder_path', type=str, default='model_params/encoder_bpe_40000.json')
	parser.add_argument('--bpe_path', type=str, default='model_params/vocab_40000.bpe')
	parser.add_argument('--task_path', type=str)

	args = parser.parse_args()
	verbose = args.verbose
	if verbose:
		print(args)

	task_path = args.task_path
	with open(task_path, 'r') as task_file:
		task = json.load(task_file)
	validate_task(task)

	set_seed(args.seed)
	device = get_device(verbose)

	text_encoder = TextEncoder(args.encoder_path, args.bpe_path)

	train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(task, text_encoder, args.test_split, args.validation_split, args.batch_size, device, verbose)

	class dotdict(dict):
	    """dot.notation access to dictionary attributes"""
	    __getattr__ = dict.get
	    __setattr__ = dict.__setitem__
	    __delattr__ = dict.__delitem__

	DEFAULT_CONFIG = dotdict({
		'n_embd': 768,
	    'n_head': 12,
	    'n_layer': 12,
	    'embd_pdrop': 0.1,
	    'attn_pdrop': 0.1,
	    'resid_pdrop': 0.1,
	    'afn': 'gelu',
	    'clf_pdrop': 0.1
	    })

	sequence_dim = train_dataloader.dataset.instances.shape[-1]
	vocab_size = len(text_encoder.encoder) + sequence_dim

	dh_model = DoubleHeadModel(DEFAULT_CONFIG, text_encoder.classify_token, task['task_type'], vocab_size, sequence_dim)

	for x, m, y in train_dataloader:
		x = x.view(-1, sequence_dim)
		m = m.view(-1, sequence_dim)
		print(x.shape)
		print(m.shape)
		print(y.shape)
		break

	#TODO: add positional encodings
	#TODO: calculate sequence_dim from both train and test
	#TODO: add number of classes to schema for document classification