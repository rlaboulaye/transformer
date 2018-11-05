import numpy as np
import torch
from torch import nn


class LSTMOptimizer(nn.Module):

	def __init__(self, module, hidden_size=20, num_layers=1, momentum=0):
		super(LSTMOptimizer, self).__init__()

		self.set_module(module)
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

		shapes = [param.shape for param in self.local_module._parameters.values() if param is not None]
		input_and_output_size = np.max([np.sum(shape) for shape in shapes])
		shape = (np.sum([np.prod(shape) for shape in shapes]), 1)
		self.theta_0 = nn.Parameter(torch.zeros(shape, device=self.device).normal_(0, np.sqrt(2. / input_and_output_size)))

		self.reset_state()

	def set_module(self, module):
		self.local_module = module

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
		self.theta_0 = self.theta_0.to(device)
		self.f_tm1 = self.f_tm1.to(device)
		self.i_tm1 = self.i_tm1.to(device)
		self.theta_tm1 = self.theta_tm1.to(device)
		self.delta_tm1 = self.delta_tm1.to(device)
		if self.state_tm1 is not None:
			self.state_tm1 = (self.state_tm1[0].to(device), self.state_tm1[1].to(device))
		return self

	def reset_state(self, learn_initialization=True):
		shape = (np.sum([np.prod(param.shape) for param in self.local_module._parameters.values() if param is not None]), 1)
		self.f_tm1 = torch.ones(shape, device=self.device)
		self.i_tm1 = torch.zeros(shape, device=self.device)
		self.delta_tm1 = torch.zeros(shape, device=self.device)
		self.state_tm1 = None
		if learn_initialization:
			self.theta_tm1 = torch.cat([param.view(-1,1) for param in self.local_module._parameters.values() if param is not None], dim=0)
		else:
			self.theta_tm1 = torch.cat([param.view(-1,1) for param in self.local_module._parameters.values() if param is not None], dim=0).clone().detach()

	def initialize_params(self):
		shapes = [param.shape for param in self.local_module._parameters.values() if param is not None]
		parameters = self.theta_0.split([np.prod(shape) for shape in shapes])
		for parameter_index, parameter_name in enumerate([parameter_name for parameter_name in self.local_module._parameters if self.local_module._parameters[parameter_name] is not None]):
			self.local_module._parameters[parameter_name] = parameters[parameter_index].view(shapes[parameter_index])

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

	def update_rule(self, grad_t, loss_t):
		batch_size = grad_t.shape[0]
		preprocessed_grad_t = self.preprocess(grad_t.view(-1)).view(1, batch_size, -1)
		preprocessed_loss_t = self.preprocess(loss_t.view(-1)).view(1, batch_size, -1)
		if self.state_tm1 is None:
			output_t, state_t = self.lstm(torch.cat([preprocessed_grad_t, preprocessed_loss_t], dim=-1))
		else:
			output_t, state_t = self.lstm(torch.cat([preprocessed_grad_t, preprocessed_loss_t], dim=-1), self.state_tm1)
		f_t = self.activation(self.W_theta(torch.cat([output_t.view(batch_size, -1), self.theta_tm1, self.f_tm1], dim=-1)))
		i_t = self.activation(self.W_grad(torch.cat([output_t.view(batch_size, -1), self.theta_tm1, self.i_tm1], dim=-1)))
		delta_t = self.momentum * self.delta_tm1 - i_t * grad_t
		theta_t = f_t * self.theta_tm1 + delta_t
		self.f_tm1 = f_t
		self.i_tm1 = i_t
		self.theta_tm1 = theta_t
		self.delta_tm1 = delta_t
		self.state_tm1 = state_t
		return theta_t

	def forward(self, module_with_grads, loss_t):
		shapes = []
		gradients = []
		for parameter_name in [parameter_name for parameter_name in module_with_grads._parameters if module_with_grads._parameters[parameter_name] is not None]:
			parameter = module_with_grads._parameters[parameter_name]
			shapes.append(parameter.shape)
			gradients.append(parameter.grad.clone().detach().view(-1, 1))
		grad_t = torch.cat(gradients, dim=0)
		param_t = self.update_rule(grad_t, grad_t.new_full(grad_t.shape, loss_t.item()))
		parameters = param_t.split([np.prod(shape) for shape in shapes])
		for parameter_index, parameter_name in enumerate(module_with_grads._parameters):
			self.local_module._parameters[parameter_name] = parameters[parameter_index].view(shapes[parameter_index])
