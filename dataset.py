import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):

	def __init__(self, device, vocab_size, instances, masks, targets=None):
		self.device = device
		self.instances = instances
		self.masks = masks
		self.targets = targets
		sequence_dim = self.instances.shape[-1]
		self.X_positions = np.arange(vocab_size, vocab_size + sequence_dim)

	def __len__(self):
		return self.instances.shape[0]

	def __getitem__(self, index):
		# M is a mask
		X_tokens = self.instances[index]
		X_value = np.zeros(X_tokens.shape + (2,))
		X_value[:, :, 0] = X_tokens
		X_value[:, :, 1] = self.X_positions
		X = torch.tensor(X_value, dtype=torch.int64, device=self.device)
		M = torch.tensor(self.masks[index], dtype=torch.int64, device=self.device)
		Y = torch.tensor(self.targets[index], dtype=torch.int64, device=self.device)
		return X, M, Y