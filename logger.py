import os
import json

from matplotlib import pyplot as plt


class Logger(object):

	def __init__(self, task_name, num_scores_per_epoch=1, default_accuracy=.5):
		self.task_name = task_name
		self.results_directory = 'results/{}'.format(self.task_name)
		self.results = {
			'train_losses': [],
			'train_accuracies': [],
			'validation_losses': [],
			'validation_accuracies': [],
			'test_loss': 0,
			'test_accuracy': 0,
			'num_scores_per_epoch': num_scores_per_epoch,
			'default_accuracy': default_accuracy
		}
		if not os.path.exists(self.results_directory):
			os.makedirs(self.results_directory)

	def load(self, file_path):
		self.results_directory, task_file_name = os.path.split(file_path)
		self.task_name = task_file_name.strip('.json')
		with open(file_path, 'r') as file_obj:
			self.results = json.load(file_obj)

	def log(self):
		with open('{}/results.json', 'w').format(self.results_directory) as file_obj:
			json.dump(file_obj)

	def plot(self):
		plt.figure()
		plt.title('Loss')
		plt.plot(self.results['train_losses'], label='train')
		plt.plot(self.results['validation_losses'], label='validate')
		plt.legend()
		plt.savefig('{}/loss.png'.format(self.results_directory))
		plt.close()

		plt.figure()
		plt.title('Accuracy')
		plt.plot(self.results['train_accuracies'], label='train')
		plt.plot(self.results['validation_accuracies'], label='validate')
		plt.plot([default_accuracy] * len(self.results['train_accuracies']), label='default')
		plt.legend()
		plt.savefig('{}/accuracy.png'.format(self.results_directory))
		plt.close()
