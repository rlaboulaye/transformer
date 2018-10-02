from tqdm import tqdm
import numpy as np
import pandas as pd


def get_dataloaders(task, text_encoder, verbose):
	dataframe = load_dataframe(task['train_file_path'])
	documents_dataframe = create_documents(dataframe, task['document_list'], task['task_type'], text_encoder, verbose)
	max_sequence_length = max([documents_dataframe[column].apply(lambda x: len(x)).max() for column in documents_dataframe.columns])
	document_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
	mask_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(np.ones(len(x)), (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
	document_matrix = np.concatenate([document_matrix.reshape(-1, 1, max_sequence_length) for document_matrix in document_matrices], axis=1)
	mask_matrix = np.concatenate([mask_matrix.reshape(-1, 1, max_sequence_length) for mask_matrix in mask_matrices], axis=1)
	# train validate test split
	document_matrix = document_matrix.reshape(-1, max_sequence_length)
	mask_matrix = mask_matrix.reshape(-1, max_sequence_length)
	print([text_encoder.decoder[token] for token in document_matrix[4]])
	print([text_encoder.decoder[token] for token in document_matrix[5]])

def load_dataframe(path):
	return pd.read_csv(path)

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
		assert(documents_dataframe.shape[1] == 1)
		return documents_dataframe.progress_apply(lambda x: [text_encoder.start_token] + x + [text_encoder.classify_token], axis=1)

def split_dataset(X, Y, test_split=.2, validation_split=.2):
	assert(X.shape[0] == Y.shape[0])
	permutation = np.random.permutation(X.shape[0])
	X_permutated = X[permutation]
	Y_permutated = Y[permutation]
	train_val_end_index = round(X.shape[0] * float(1 - test_split))
	train_end_index = round(train_val_end_index * float(1 - validation_split))
	train_range = range(train_end_index)
	validation_range = range(train_end_index, train_val_end_index)
	test_range = range(train_val_end_index, X.shape[0])
	train_set = [X[train_range], Y[train_range]]
	validation_set = [X[validation_range], Y[validation_range]]
	test_set = [X[test_range], Y[test_range]]
	return train_set, validation_set, test_set
