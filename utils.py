import sys
import random
import json

from tqdm import tqdm
import numpy as np
import torch

from jsonschema import validate
from jsonschema.exceptions import ValidationError


def get_iterator(obj, verbose):
	if verbose:
		return tqdm(obj, ncols=80)
	return iter(obj)

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def get_device(verbose=True):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if verbose:
		print("device: {}".format(device))
	return device

def validate_task(task, task_schema_path='schema/task_schema.json'):
	with open(task_schema_path, 'r') as task_schema_file:
		task_schema = json.load(task_schema_file)
	try:
		validate(task, task_schema)
	except ValidationError as err:
		sys.exit('EXCEPTION: THE TASK FAILED TO VALIDATE AGAINST THE TASK SCHEMA.\n\n{}'.format(err))