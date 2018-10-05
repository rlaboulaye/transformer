from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils import data

from .dataset import Dataset


def get_dataloaders(task, text_encoder, test_split, validation_split, batch_size, device, verbose):
	train_file = task['train_file']
	train_dataframe = load_dataframe(train_file['file_path'], train_file['file_type'], train_file['file_header'])
	train_document_matrix, train_mask_matrix = get_document_matrix(train_dataframe, task['document_list'], task['task_type'], text_encoder, verbose)
	train_matrices = (train_document_matrix, train_mask_matrix)
	if 'target' in task:
		train_matrices += (train_dataframe[train_dataframe.columns[task['target']['column_index']]].values,)
	if 'test_file_path' in task:
		test_file = task['test_file']
		test_dataframe = load_dataframe(test_file['file_path'], test_file['file_type'], test_file['file_header'])
		test_document_matrix, test_mask_matrix = get_document_matrix(test_dataframe, task['document_list'], task['task_type'], text_encoder, verbose)
		test_matrices = (test_document_matrix, test_mask_matrix)
		if 'target' in task:
			test_matrices += (test_dataframe[test_dataframe.columns[task['target']['column_index']]].values,)
		train_matrices, validation_matrices = split_data(train_matrices, validation_split)
	else:
		train_val_matrices, test_matrices = split_data(train_matrices, test_split)
		train_matrices, validation_matrices = split_data(train_val_matrices, validation_split)
	vocab_size = len(text_encoder.encoder)
	train_set = Dataset(device, vocab_size, *train_matrices)
	validation_set = Dataset(device, vocab_size, *validation_matrices)
	test_set = Dataset(device, vocab_size, *test_matrices)
	data_params = {
			'batch_size': batch_size,
			'shuffle': True
	}
	return data.DataLoader(train_set, **data_params), data.DataLoader(validation_set, **data_params), data.DataLoader(test_set, **data_params)

def load_dataframe(path, file_type, has_header):
	if file_type == 'csv':
		separator = ','
	elif file_type == 'tsv':
		separator = sep='\t'
	else:
		raise NotImplementedError('Cannot load {} file type'.format(file_type))
	if has_header:
		return pd.read_csv(path, sep=separator, header=0)
	else:
		return pd.read_csv(path, sep=separator)

def get_document_matrix(dataframe, document_list, task_type, text_encoder, verbose):
	documents_dataframe = create_documents(dataframe, document_list, task_type, text_encoder, verbose)
	max_sequence_length = max([documents_dataframe[column].apply(lambda x: len(x)).max() for column in documents_dataframe.columns])
	document_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
	mask_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(np.ones(len(x)), (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
	document_matrix = np.concatenate([document_matrix.reshape(-1, 1, max_sequence_length) for document_matrix in document_matrices], axis=1)
	mask_matrix = np.concatenate([mask_matrix.reshape(-1, 1, max_sequence_length) for mask_matrix in mask_matrices], axis=1)
	return document_matrix, mask_matrix

def create_documents(dataframe, document_list, task_type, text_encoder, verbose):
	encoded_documents = []
	for document_index, document in enumerate(document_list):
		tqdm.pandas(disable=not verbose, ncols=150, desc='Creating document {} of {} for each instance'.format(document_index + 1, len(document_list)))
		document_dataframe = dataframe[dataframe.columns[document['column_indices']]].progress_apply(lambda x: ' '.join(x), axis=1)
		tqdm.pandas(disable=not verbose, ncols=150, desc='Encoding document {} of {} for each instance'.format(document_index + 1, len(document_list)))
		encoded_documents.append(document_dataframe.progress_apply(text_encoder.encode))
	documents_dataframe = pd.concat(encoded_documents, axis=1)
	tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to {} document(s) for each instance'.format(documents_dataframe.shape[1] - 1))
	if task_type == 'DocumentSimilarity' or task_type == 'QuestionAnswering':
		assert(documents_dataframe.shape[1] == 2)
		return documents_dataframe.progress_apply(lambda x: [text_encoder.start_token] + x[documents_dataframe.columns[0]] + [text_encoder.delimeter_token] + x[documents_dataframe.columns[1]] + [text_encoder.classify_token], axis=1)
	elif task_type == 'MultipleChoice':
		assert(documents_dataframe.shape[1] > 1)
		multiple_choice_documents = []
		common_column_name = documents_dataframe.columns[0]
		for choice_column_name in documents_dataframe.columns[1:]:
			multiple_choice_documents.append(documents_dataframe[[common_column_name, choice_column_name]].progress_apply(lambda x: [text_encoder.start_token] + x[common_column_name] + [text_encoder.delimeter_token] + x[choice_column_name] + [text_encoder.classify_token], axis=1))
		return pd.concat(multiple_choice_documents, axis=1)
	else:
		tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to 1 document for each instance')
		assert(documents_dataframe.shape[1] == 1)
		return documents_dataframe.progress_apply(lambda x: [text_encoder.start_token] + x + [text_encoder.classify_token], axis=1)

def split_data(matrices, split=.2):
	# Check that all matrices have the same number of rows
	assert(1 == len(set([matrix.shape[0] for matrix in matrices])))
	num_rows = matrices[0].shape[0]
	permutation = np.random.permutation(num_rows)
	split1_end_index = round(num_rows * float(1 - split))
	split1_range = range(split1_end_index)
	split2_range = range(split1_end_index, num_rows)
	split1_matrices = ()
	split2_matrices = ()
	for matrix in matrices:
		permuted_matrix = matrix[permutation]
		split1_matrix = permuted_matrix[split1_range]
		split2_matrix = permuted_matrix[split2_range]
		split1_matrices += (split1_matrix,)
		split2_matrices += (split2_matrix,)
	return split1_matrices, split2_matrices
