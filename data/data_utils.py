from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
import math

from .dataset import Dataset


def get_dataloaders(task, text_encoder, test_split, validation_split, batch_size, device, verbose, max_sequence_length=None):

	train_file = task['train_file']
	train_val_dataframe = load_dataframe(train_file['file_path'], train_file['file_type'], train_file['file_header'])
	if 'test_file' in task:
		test_file = task['test_file']
		test_dataframe = load_dataframe(test_file['file_path'], test_file['file_type'], test_file['file_header'])
	else:
		train_val_dataframe, test_dataframe = split_dataframe(train_val_dataframe, test_split)
	train_dataframe, validation_dataframe = split_dataframe(train_val_dataframe, validation_split)

	train_documents_dataframe = create_documents(train_dataframe, task["document_list"], task["task_type"], text_encoder, verbose, max_sequence_length)
	validation_documents_dataframe = create_documents(validation_dataframe, task["document_list"], task["task_type"], text_encoder, verbose, max_sequence_length)
	test_documents_dataframe = create_documents(test_dataframe, task["document_list"], task["task_type"], text_encoder, verbose, max_sequence_length)

	if max_sequence_length is None:
		max_sequence_length = max(
			max([train_documents_dataframe[column].apply(lambda x: len(x)).max() for column in train_documents_dataframe.columns]),
			max([validation_documents_dataframe[column].apply(lambda x: len(x)).max() for column in validation_documents_dataframe.columns]),
			max([test_documents_dataframe[column].apply(lambda x: len(x)).max() for column in test_documents_dataframe.columns]))

	train_document_matrix, train_mask_matrix = get_document_matrix(train_documents_dataframe, max_sequence_length)
	train_matrices = (train_document_matrix, train_mask_matrix)
	validation_document_matrix, validation_mask_matrix = get_document_matrix(validation_documents_dataframe, max_sequence_length)
	validation_matrices = (validation_document_matrix, validation_mask_matrix)
	test_document_matrix, test_mask_matrix = get_document_matrix(test_documents_dataframe, max_sequence_length)
	test_matrices = (test_document_matrix, test_mask_matrix)

	# what is target_encoders for?
	if 'target' in task:
		train_target_matrix, target_encoders = get_target_matrix(train_dataframe, task['target']['column_indices'], task['task_type'])
		train_matrices += (train_target_matrix,)
		validation_target_matrix, target_encoders = get_target_matrix(validation_dataframe, task['target']['column_indices'], task['task_type'])
		validation_matrices += (validation_target_matrix,)
		test_target_matrix, target_encoders = get_target_matrix(test_dataframe, task['target']['column_indices'], task['task_type'])
		test_matrices += (test_target_matrix,)

	vocab_size = len(text_encoder.encoder)

	train_set = Dataset(device, task['task_type'], vocab_size, *train_matrices)
	validation_set = Dataset(device, task['task_type'], vocab_size, *validation_matrices)
	test_set = Dataset(device, task['task_type'], vocab_size, *test_matrices)
	data_params = {
			'batch_size': batch_size,
			'shuffle': True
	}
	return data.DataLoader(train_set, **data_params), data.DataLoader(validation_set, **data_params), data.DataLoader(test_set, **data_params)


def load_dataframe(path, file_type, has_header):
	if file_type == 'csv':
		separator = ','
	elif file_type == 'tsv':
		separator = '\t'
	else:
		raise NotImplementedError('Cannot load {} file type'.format(file_type))
	if has_header:
		return pd.read_csv(path, sep=separator, header=0)
	else:
		return pd.read_csv(path, sep=separator)

def get_target_matrix(dataframe, target_indices, task_type, encoders=None):
	if task_type is "DocumentSimilarity":
		return dataframe[dataframe.columns[target_indices]].values
	else:
		targets = []
		if encoders is None:
			encoders = ()
			for index in target_indices:
				target_col = dataframe[dataframe.columns[index]]
				encoder = LabelEncoder()
				targets.append(encoder.fit_transform(target_col).reshape(-1,1))
				encoders += (encoder,)
		else:
			for encoder, index in zip(encoders, target_indices):
				targets.append(encoder.transform(dataframe[dataframe.columns[index]]).reshape(-1,1))
		target_matrix = np.concatenate(targets, axis=1)
		if target_matrix.shape[1] == 1:
			target_matrix = target_matrix.reshape(-1)
		return target_matrix, encoders

def get_document_matrix(documents_dataframe, max_sequence_length):
	document_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
	mask_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(np.ones(len(x)), (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
	document_matrix = np.concatenate([document_matrix.reshape(-1, 1, max_sequence_length) for document_matrix in document_matrices], axis=1)
	mask_matrix = np.concatenate([mask_matrix.reshape(-1, 1, max_sequence_length) for mask_matrix in mask_matrices], axis=1)
	return document_matrix, mask_matrix

def create_documents(dataframe, document_list, task_type, text_encoder, verbose, max_sequence_dim):
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
		num_tokens = 3
		doc1_length = math.ceil((max_sequence_dim - num_tokens) / 2) if max_sequence_dim is not None else None
		doc2_length = math.floor((max_sequence_dim - num_tokens) / 2) if max_sequence_dim is not None else None
		return documents_dataframe.progress_apply(lambda x: [text_encoder.start_token] + x[documents_dataframe.columns[0]][:doc1_length] + [text_encoder.delimeter_token] + x[documents_dataframe.columns[1]][:doc2_length] + [text_encoder.classify_token], axis=1)
	elif task_type == 'MultipleChoice':
		assert(documents_dataframe.shape[1] > 1)
		multiple_choice_documents = []
		common_column_name = documents_dataframe.columns[0]

		if max_sequence_dim is not None:
			num_tokens = 4
			max_len = max_sequence_dim - num_tokens
			documents_dataframe['scale'] = pd.Series(documents_dataframe.apply(lambda x: max_len / (len(x[0]) + max([len(y) for y in x[1:]])), axis=1), index=documents_dataframe.index)
			scale_column_name = documents_dataframe.columns[-1]
			for choice_column_name in documents_dataframe.columns[1:-1]:
				multiple_choice_documents.append(documents_dataframe[[scale_column_name, common_column_name, choice_column_name]].progress_apply(lambda x:
																																				 [text_encoder.start_token] +
																																				 x[common_column_name][:math.floor(len(x[common_column_name]) * x[scale_column_name])] +
																																				 [text_encoder.delimeter_token] +
																																				 x[choice_column_name][:max_len - math.floor(len(x[common_column_name]) * x[scale_column_name])] +
																																				 [text_encoder.classify_token], axis=1))
		else:
			for choice_column_name in documents_dataframe.columns[1:]:
				multiple_choice_documents.append(documents_dataframe[[common_column_name, choice_column_name]].progress_apply(lambda x: [text_encoder.start_token] + x[common_column_name] + [text_encoder.delimeter_token] + x[choice_column_name] + [text_encoder.classify_token], axis=1))

		return pd.concat(multiple_choice_documents, axis=1)
	else:
		tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to 1 document for each instance')
		assert(documents_dataframe.shape[1] == 1)
		num_tokens = 2
		doc_length = max_sequence_dim - num_tokens
		return documents_dataframe.progress_apply(lambda x: [text_encoder.start_token] + x[:doc_length] + [text_encoder.classify_token], axis=1)

def split_dataframe(dataframe, split=.2):
	split_df2 = dataframe.sample(frac=split, replace=False)
	split_df1 = dataframe.drop(split_df2.index)
	return split_df1, split_df2
