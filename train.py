import json
import argparse

from text_encoder import TextEncoder
from utils import set_seed, get_device, validate_task
from data_utils import get_dataloaders


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
	n_vocab = len(text_encoder.encoder)

	train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(task, text_encoder, args.test_split, args.validation_split, args.batch_size, device, verbose)

	for x, m, y in train_dataloader:
		print(x.shape)
		print(m.shape)
		print(y.shape)
		print()