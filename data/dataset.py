import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):

	def __init__(self, device, target_type, vocab_size, instances, masks, targets=None, max_position_encoding=512):
		self.device = device
		self.task_type = target_type
		self.instances = instances
		self.masks = masks
		self.targets = targets
		self.sequence_dim = self.instances.shape[-1]
		self.max_position_encoding = min(self.sequence_dim, max_position_encoding)
		self.X_positions = np.arange(vocab_size, vocab_size + self.max_position_encoding)
		while self.X_positions.shape[0] < self.sequence_dim:
			self.X_positions = np.concatenate([self.X_positions,
				np.arange(vocab_size, vocab_size + min(self.sequence_dim - self.X_positions.shape[0], max_position_encoding))], 0)
		if target_type == "regression":
			self.target_dtype = torch.float32
		else:
			self.target_dtype = torch.int64

	def __len__(self):
		return self.instances.shape[0]

	def __getitem__(self, index):
		# M is a mask
		X_tokens = self.instances[index]
		X_value = np.zeros(X_tokens.shape + (2,))
		X_value[:, :, 0] = X_tokens
		X_value[:, :, 1] = self.X_positions
		X = torch.tensor(X_value, dtype=torch.int64, device=self.device)
		M = torch.tensor(self.masks[index], dtype=torch.float32, device=self.device)
		if self.targets is None:
			return X, M
		else:
			Y = torch.tensor(self.targets[index], dtype=self.target_dtype, device=self.device)
			return X, M, Y