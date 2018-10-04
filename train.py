import json
import argparse

from utils import set_seed, get_device, validate_task
from data.text_encoder import TextEncoder
from data.data_utils import get_dataloaders
from model.double_head_model import DoubleHeadModel


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--batch_size', type=int, default=8)
	# TODO Rename arguments
	parser.add_argument('--n_embd', type=int, default=768)
	parser.add_argument('--n_head', type=int, default=12)
	parser.add_argument('--n_layer', type=int, default=12)
	parser.add_argument('--embd_pdrop', type=float, default=.1)
	parser.add_argument('--attn_pdrop', type=float, default=.1)
	parser.add_argument('--resid_pdrop', type=float, default=.1)
	parser.add_argument('--clf_pdrop', type=float, default=.1)
	parser.add_argument('--afn', type=str, choices=['relu', 'swish', 'gelu'], default='gelu')
	#
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

	sequence_dim = train_dataloader.dataset.sequence_dim
	vocab_size = len(text_encoder.encoder) + sequence_dim

	dh_model = DoubleHeadModel(args, text_encoder.classify_token, task['task_type'], vocab_size, sequence_dim)
	dh_model.to(device)

	for x, m, y in train_dataloader:
		lm_logits, task_logits = dh_model(x)
		print(lm_logits.shape)
		print(task_logits.shape)
		break

	#TODO: calculate sequence_dim from both train and test
	#TODO: add number of classes to schema for document classification