import torch
from torch.utils import data


class Dataset(data.Dataset):

	def __init__(self, device):
		self.device = device