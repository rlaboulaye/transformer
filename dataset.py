import torch
from torch.utils import data


class Dataset(data.Dataset):

	def __init__(self, device, instances, masks, targets=None):
		self.device = device
		self.instances = instances
		self.masks = masks
		self.targets = targets

	def __len__(self):
		return self.instances.shape[0]

	def __getitem__(self, index):
		# M is a mask
		X = torch.tensor(self.instances[index], dtype=torch.int64, device=self.device)
		M = torch.tensor(self.masks[index], dtype=torch.int64, device=self.device)
		Y = torch.tensor(self.targets[index], dtype=torch.int64, device=self.device)
		return X, M, Y