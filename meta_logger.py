import os
import json
import datetime

import numpy as np
from matplotlib import pyplot as plt


class MetaLogger(object):

    def __init__(self, meta_config):
        self.results_directory = os.path.join('meta_results', str(datetime.datetime.now()))
        self.results = {
            'train_losses': [],
            'train_accuracies': [],
            'validation_losses': [],
            'validation_accuracies': [],
            'baseline_test_loss': 0,
            'baseline_test_accuracy': 0,
            'sgd_test_loss': 0,
            'sgd_test_accuracy': 0,
            'adam_test_loss': 0,
            'adam_test_accuracy': 0,
            'stacked_optimizer_test_loss': 0,
            'stacked_optimizer_test_accuracy': 0,
            'config': meta_config
        }
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def load(self, file_path):
        self.results_directory, _ = os.path.split(file_path)
        with open(file_path, 'r') as file_obj:
            self.results = json.load(file_obj)

    def log(self):
        with open('{}/results.json'.format(self.results_directory), 'w') as file_obj:
            json.dump(self.results, file_obj, indent=4)

    def plot(self):
        plt.figure()
        plt.title('Loss')
        plt.xlabel('Meta Epochs')
        plt.ylabel('Loss')
        plt.plot(self.results['train_losses'], label='train')
        plt.plot(self.results['validation_losses'], label='validate')
        plt.legend()
        plt.savefig('{}/loss.png'.format(self.results_directory))
        plt.close()

        plt.figure()
        plt.title('Accuracy')
        plt.xlabel('Meta Epochs')
        plt.ylabel('Accuracy')
        plt.plot(self.results['train_accuracies'], label='train')
        plt.plot(self.results['validation_accuracies'], label='validate')
        plt.legend()
        plt.savefig('{}/accuracy.png'.format(self.results_directory))
        plt.close()

        plt.figure()
        plt.title('Test Losses')
        plt.ylabel('Mean Test Loss')
        x_labels = ('Baseline', 'SGD', 'Adam', 'Stacked Optimizer')
        x_pos = np.arange(len(x_labels))
        performance = [self.results['{}_test_loss'.format('_'.join(label.lower().split(' ')))] for label in x_labels]
        plt.bar(x_pos, performance, align='center', alpha=0.5)
        plt.xticks(x_pos, x_labels)
        plt.savefig('{}/test_loss.png'.format(self.results_directory))
        plt.close()

        plt.figure()
        plt.title('Test Accuracies')
        plt.ylabel('Mean Test Accuracy')
        x_labels = ('Baseline', 'SGD', 'Adam', 'Stacked Optimizer')
        x_pos = np.arange(len(x_labels))
        performance = [self.results['{}_test_accuracy'.format('_'.join(label.lower().split(' ')))] for label in x_labels]
        plt.bar(x_pos, performance, align='center', alpha=0.5)
        plt.xticks(x_pos, x_labels)
        plt.savefig('{}/test_accuracy.png'.format(self.results_directory))
        plt.close()
