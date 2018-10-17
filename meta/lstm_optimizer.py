import numpy as np
import torch
from torch import nn

from local_model import LocalModel


class LSTMOptimizer(nn.Module):

	def __init__(self, module, hidden_size=20, num_layers=1, momentum=0):
		super(LSTMOptimizer, self).__init__()

		self.device = 'cpu'
		self.local_module = module
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

		self.theta_0 = nn.Module()
		for parameter_name in self.local_module._parameters:
			parameter = self.local_module._parameters[parameter_name]
			if len(parameter.shape) == 1:
				setattr(self.theta_0, parameter_name, nn.Parameter(torch.zeros(parameter.shape, device=self.device).normal_(0, .1)))
			else:
				setattr(self.theta_0, parameter_name, nn.Parameter(torch.zeros(parameter.shape, device=self.device).normal_(0, np.sqrt(2. / sum(parameter.shape)))))

		self.reset_state()

	def initialize_optimizer_params(self, window=.01):
		for param in self.lstm.parameters():
			param.data.uniform_(-window, window)
		self.W_theta.weight.data.uniform_(-window, window)
		self.W_theta.bias.data.uniform_(11., 12.)
		self.W_grad.weight.data.uniform_(-window, window)
		self.W_grad.bias.data.uniform_(-3.5, -4.5)

	def to(self, device):
		self.device = device
		super(LSTMOptimizer, self).to(device)
		for parameter_name in self.theta_tm1:
			self.f_tm1[parameter_name] = self.f_tm1[parameter_name].to(device)
			self.i_tm1[parameter_name] = self.i_tm1[parameter_name].to(device)
			self.theta_tm1[parameter_name] = self.theta_tm1[parameter_name].to(device)
			self.delta_tm1[parameter_name] = self.delta_tm1[parameter_name].to(device)
			if self.state_tm1[parameter_name] is not None:
				self.state_tm1[parameter_name] = (self.state_tm1[parameter_name][0].to(device), self.state_tm1[parameter_name][1].to(device))
		return self

	def reset_state(self):
		self.f_tm1 = {}
		self.i_tm1 = {}
		self.theta_tm1 = {}
		self.delta_tm1 = {}
		self.state_tm1 = {}
		for parameter_name in self.local_module._parameters:
			parameter = self.local_module._parameters[parameter_name]
			self.f_tm1[parameter_name] = torch.ones(parameter.shape, device=self.device)
			self.i_tm1[parameter_name] = torch.zeros(parameter.shape, device=self.device)
			self.theta_tm1[parameter_name] = getattr(self.theta_0, parameter_name)
			self.delta_tm1[parameter_name] = torch.zeros(parameter.shape, device=self.device)
			self.state_tm1[parameter_name] = None

	def initialize_params(self):
		for parameter_name in self.local_module._parameters:
			self.local_module._parameters[parameter_name] = getattr(self.theta_0, parameter_name)

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

	def update_rule(self, grad_t, loss_t, parameter_name):
		batch_size = grad_t.shape[0]
		f_tm1 = self.f_tm1[parameter_name].view(-1, 1)
		i_tm1 = self.i_tm1[parameter_name].view(-1, 1)
		theta_tm1 = self.theta_tm1[parameter_name].view(-1, 1)
		delta_tm1 = self.delta_tm1[parameter_name].view(-1, 1)
		state_tm1 = self.state_tm1[parameter_name]
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
		self.f_tm1[parameter_name] = f_t.view(self.local_module._parameters[parameter_name].shape)
		self.i_tm1[parameter_name] = i_t.view(self.local_module._parameters[parameter_name].shape)
		self.theta_tm1[parameter_name] = theta_t.view(self.local_module._parameters[parameter_name].shape)
		self.delta_tm1[parameter_name] = delta_t.view(self.local_module._parameters[parameter_name].shape)
		self.state_tm1[parameter_name] = state_t
		return theta_t

	def forward(self, module_with_grads, loss_t):
		for parameter_name in module_with_grads._parameters:
			parameter = module_with_grads._parameters[parameter_name]
			grad_t = parameter.grad.clone().detach().view(-1, 1)
			self.local_module._parameters[parameter_name] = self.update_rule(grad_t, grad_t.new_full(grad_t.shape, loss_t.item()), parameter_name).view(parameter.shape)
