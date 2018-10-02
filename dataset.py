import pandas as pd
import torch
from torch.utils import data


class Dataset(data.Dataset):

	def __init__(self, task, device):
		self.device = device
		dataframe = self.load_dataframe(task['train_file_path'])
		documents_dataframe = self.create_documents(dataframe, task['document_list'], task['task_type'])
		print(documents_dataframe)
		print(documents_dataframe.shape)
		print(documents_dataframe.iloc[3,0])
		print(documents_dataframe.iloc[3,1])
		print(documents_dataframe.iloc[4,0])

	def load_dataframe(self, path):
		return pd.read_csv(path)

	def create_documents(self, dataframe, document_list, task_type):
		encoded_documents = []
		for document in document_list:
			document_dataframe = dataframe[dataframe.columns[document['column_indices']]].apply(lambda x: ' '.join(x), axis=1)
			encoded_documents.append(document_dataframe.apply(lambda x: text_encoder.encode(x), axis=1))
			# TODO: encode documents
		documents_dataframe = pd.concat(encoded_documents, axis=1)
		if task_type == 'DocumentSimilarity' or task_type == 'QuestionAnswering':
			assert(documents_dataframe.shape[1] == 2)
			return documents_dataframe.apply(lambda x: '<start>' + '<delimeter>'.join(x) + '<end>', axis=1)
		elif task_type == 'MultipleChoice':
			assert(documents_dataframe.shape[1] > 1)
			multiple_choice_documents = []
			common_column_name = documents_dataframe.columns[0]
			for choice_column_name in documents_dataframe.columns[1:]:
				multiple_choice_documents.append(documents_dataframe[[common_column_name, choice_column_name]].apply(lambda x: '<start>' + '<delimeter>'.join(x) + '<end>', axis=1))
			return pd.concat(multiple_choice_documents, axis=1)
		else:
			assert(documents_dataframe.shape[1] == 1)
			return documents_dataframe.apply(lambda x: '<start>' + x + '<end>', axis=1)