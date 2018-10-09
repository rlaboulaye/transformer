import torch
from torch import nn


class MetaUpdateRule(nn.Module):

	def __init__(self):
		super(WeightUpdateRule, self).__init__()
		self.weight_W = nn.Linear(x, x, bias=True)
		self.gradient_W = nn.Linear(x, x, bias=True)

	def forward(self, weights, gradients, loss):
		pass