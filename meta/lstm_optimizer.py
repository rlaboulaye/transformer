import numpy as np
import torch
from torch import nn


class LSTMOptimizer(nn.Module):

	def __init__(self, params):
		super(LSTMOptimizer, self).__init__()

		self.device = 'cpu'

		self.p = 10
		self.exp_p = np.exp(self.p)
		self.neg_exp_p = np.exp(-self.p)

		input_dim = 6
		self.W_theta = nn.Linear(input_dim, 1, bias=True)
		self.W_grad = nn.Linear(input_dim, 1, bias=True)
		self.activation = nn.Sigmoid()		

		self.theta_0 = nn.ParameterList()
		for group in params:
			self.theta_0.append(nn.Parameter(torch.zeros(group.shape)))

		self.set_params(params)

	def to(self, device):
		self.device = device
		super(LSTMOptimizer, self).to(device)
		for group_index in range(len(self.param_groups)):
			self.f_tm1[group_index] = self.f_tm1[group_index].to(device)
			self.i_tm1[group_index] = self.i_tm1[group_index].to(device)
			self.preprocessed_grad_tm1[group_index] = self.preprocessed_grad_tm1[group_index].to(device)
		return self

	def set_params(self, params):
		self.param_groups = list(params)
		self.f_tm1 = []
		self.i_tm1 = []
		self.preprocessed_grad_tm1 = []
		for group in self.param_groups:
			self.f_tm1.append(torch.ones(group.shape, device=self.device))
			self.i_tm1.append(torch.zeros(group.shape, device=self.device))
			self.preprocessed_grad_tm1.append(torch.zeros(group.shape + (2,), device=self.device))

	def initialize_params(self, params):
		for group_index, group in enumerate(params):
			group.data = self.theta_0[group_index].data

	def zero_grad(self):
		for group in self.param_groups:
			if group.grad is not None:
				group.grad.detach_()
				group.grad.zero_()

	def preprocess(self, x):
		return torch.cat([self.preprocess_1(x).view(x.shape + (1,)), self.preprocess_2(x).view(x.shape + (1,))], dim=-1)

	def preprocess_1(self, x):
		abs_x = x.abs()
		condition_1 = abs_x.ge(self.neg_exp_p)
		condition_2 = abs_x.lt(self.neg_exp_p)
		x_1 = x[condition_1]
		x_2 = x[condition_2]
		x_1 = x_1.abs().log() / self.p
		x_2 = x_2.new_full(x_2.shape, -1)
		z = x.new_zeros(x.shape)
		z[condition_1] = x_1
		z[condition_2] = x_2
		return z

	def preprocess_2(self, x):
		abs_x = x.abs()
		condition_1 = abs_x.ge(self.neg_exp_p)
		condition_2 = abs_x.lt(self.neg_exp_p)
		x_1 = x[condition_1]
		x_2 = x[condition_2]
		x_1 = x_1.sign()
		x_2 = x_2 * self.exp_p
		z = x.new_zeros(x.shape)
		z[condition_1] = x_1
		z[condition_2] = x_2
		return z

	def update(self, theta_tm1, grad_t, loss, group_index):
		f_tm1 = self.f_tm1[group_index].view(-1, 1)
		i_tm1 = self.i_tm1[group_index].view(-1, 1)
		preprocessed_grad_tm1 = self.preprocessed_grad_tm1[group_index].view(-1, 1)
		preprocessed_grad_t = self.preprocess(grad_t.view(-1))
		preprocessed_loss = self.preprocess(loss.view(-1))
		f_t = self.activation(self.W_theta(torch.cat([preprocessed_grad_t, preprocessed_loss, theta_tm1, f_tm1], dim=-1)))
		i_t = self.activation(self.W_grad(torch.cat([preprocessed_grad_t, preprocessed_loss, theta_tm1, i_tm1], dim=-1)))
		theta_t = f_t * theta_tm1 - i_t * grad_t
		self.f_tm1[group_index] = f_t.view(self.f_tm1[group_index].shape)
		self.i_tm1[group_index] = i_t.view(self.i_tm1[group_index].shape)
		# not yet being used
		self.preprocessed_grad_tm1[group_index] = preprocessed_grad_t.view(self.preprocessed_grad_tm1[group_index].shape)
		return theta_t

	def forward(self, loss):
		for group_index, group in enumerate(self.param_groups):
			grad = group.grad.data.view(-1, 1)
			# need to figure out if i need to individualize loss
			group.data = self.update(group.data.view(-1, 1), grad, grad.new_full(grad.shape, loss.item()), group_index).view(group.shape)