import numpy as np
import torch
from torch import nn


class LSTMOptimizer(nn.Module):

	def __init__(self, params, hidden_size=20, num_layers=1, momentum=.1):
		super(LSTMOptimizer, self).__init__()

		self.device = 'cpu'
		self.momentum = momentum

		self.p = 10
		self.exp_p = np.exp(self.p)
		self.neg_exp_p = np.exp(-self.p)

		self.lstm = nn.LSTM(
			input_size=4,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bias=True
			)

		input_dim = hidden_size + 2
		self.activation = nn.Sigmoid()
		self.W_theta = nn.Linear(input_dim, 1, bias=True)
		self.W_grad = nn.Linear(input_dim, 1, bias=True)
		self.initialize_optimizer_params()

		self.theta_0 = nn.ParameterList()
		for group in params:
			self.theta_0.append(nn.Parameter(torch.zeros(group.shape).normal_(0, np.sqrt(2. / sum(group.shape)))))

		self.set_params(params)

	def initialize_optimizer_params(self, window=.01):
		for param in self.lstm.parameters():
			param.data.uniform_(-window, window)
		self.W_theta.weight.data.uniform_(-window, window)
		self.W_theta.bias.data.uniform_(4., 5.)
		self.W_grad.weight.data.uniform_(-window, window)
		self.W_grad.bias.data.uniform_(-4., -5.)

	def to(self, device):
		self.device = device
		super(LSTMOptimizer, self).to(device)
		for group_index in range(len(self.param_groups)):
			self.f_tm1[group_index] = self.f_tm1[group_index].to(device)
			self.i_tm1[group_index] = self.i_tm1[group_index].to(device)
			self.theta_tm1[group_index] = self.theta_tm1[group_index].to(device)
			self.delta_tm1[group_index] = self.delta_tm1[group_index].to(device)
			if self.self.state_tm1[group_index] is not None:
				self.state_tm1[group_index] = (self.state_tm1[group_index][0].to(device), self.state_tm1[group_index][1].to(device))
		return self

	def set_params(self, params):
		self.param_groups = list(params)
		self.f_tm1 = []
		self.i_tm1 = []
		self.theta_tm1 = []
		self.delta_tm1 = []
		self.state_tm1 = []
		for group_index, group in enumerate(self.param_groups):
			self.f_tm1.append(torch.ones(group.shape, device=self.device))
			self.i_tm1.append(torch.zeros(group.shape, device=self.device))
			self.theta_tm1.append(self.theta_0[group_index])
			self.delta_tm1.append(torch.zeros(group.shape, device=self.device))
			self.state_tm1.append(None)

	def initialize_params(self, params):
		for group_index, group in enumerate(params):
			group.data.copy_(self.theta_0[group_index].data)

	def update_params(self, params):
		for group_index, group in enumerate(params):
			group.data.copy_(self.theta_tm1[group_index].data)

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

	def update_rule(self, grad_t, loss_t, group_index):
		batch_size = grad_t.shape[0]
		f_tm1 = self.f_tm1[group_index].view(-1, 1)
		i_tm1 = self.i_tm1[group_index].view(-1, 1)
		theta_tm1 = self.theta_tm1[group_index].view(-1, 1)
		delta_tm1 = self.delta_tm1[group_index].view(-1, 1)
		state_tm1 = self.state_tm1[group_index]
		preprocessed_grad_t = self.preprocess(grad_t.view(-1)).view(1, batch_size, -1)
		preprocessed_loss_t = self.preprocess(loss_t.view(-1)).view(1, batch_size, -1)
		if state_tm1 is None:
			output_t, state_t = self.lstm(torch.cat([preprocessed_grad_t, preprocessed_loss_t], dim=-1))
		else:
			output_t, state_t = self.lstm(torch.cat([preprocessed_grad_t, preprocessed_loss_t], dim=-1), state_tm1)
		f_t = self.activation(self.W_theta(torch.cat([output_t.view(batch_size, -1), theta_tm1, f_tm1], dim=-1)))
		i_t = self.activation(self.W_grad(torch.cat([output_t.view(batch_size, -1), theta_tm1, i_tm1], dim=-1)))
		delta_t = self.momentum * delta_tm1 - i_t * grad_t
		theta_t = f_t * theta_tm1 + delta_t
		self.f_tm1[group_index] = f_t.view(self.f_tm1[group_index].shape)
		self.i_tm1[group_index] = i_t.view(self.i_tm1[group_index].shape)
		self.theta_tm1[group_index] = theta_t.view(self.f_tm1[group_index].shape)
		self.delta_tm1[group_index] = delta_t.view(self.f_tm1[group_index].shape)
		self.state_tm1[group_index] = state_t
		return theta_t

	def forward(self, loss_t):
		for group_index, group in enumerate(self.param_groups):
			grad_t = group.grad.detach().view(-1, 1)
			group.data.copy_(self.update_rule(grad_t, grad_t.new_full(grad_t.shape, loss_t.item()), group_index).view(group.shape).data)