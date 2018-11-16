import torch
from torch import nn

from .local_model import LocalModel
from .lstm_optimizer import LSTMOptimizer


class StackedOptimizer(nn.Module):

	def __init__(self, model, hidden_size=20, num_layers=1, momentum=0, learn_initialization_indices=[]):
		super(StackedOptimizer, self).__init__()
		self.local_model = LocalModel(model)
		self.initialize_optimizers(hidden_size, num_layers, momentum, learn_initialization_indices)

	def set_model(self, model):
		self.local_model = LocalModel(model)
		modules = [module for module in self.local_model.model.modules() if len([param for param in module._parameters.values() if param is not None and param.requires_grad]) > 0]
		for module_index, module in enumerate(modules):
			optimizer = self.optimizers[module_index]
			optimizer.set_module(modules[module_index])

	def to(self, device):
		super(StackedOptimizer, self).to(device)
		for optimizer in self.optimizers:
			optimizer.to(device)

	def initialize_optimizers(self, hidden_size, num_layers, momentum, learn_initialization_indices):
		self.optimizers = nn.ModuleList()
		modules = [module for module in self.local_model.model.modules() if len([param for param in module._parameters.values() if param is not None and param.requires_grad]) > 0]
		for module_index, module in enumerate(modules):
			learn_initialization = module_index in learn_initialization_indices
			self.optimizers.append(LSTMOptimizer(module, hidden_size, num_layers, momentum, learn_initialization=learn_initialization))

	def reset_state(self):
		for optimizer in self.optimizers:
			optimizer.reset_state()

	def initialize_params(self, model, learn_initialization_indices=[]):
		self.set_model(model)
		for module_index in learn_initialization_indices:
			optimizer = self.optimizers[module_index]
			optimizer.initialize_params()
		self.local_model.copy_params_to(model)

	def forward(self, model_with_grads, loss, learning_module_index):
		loss_t = loss.clone().detach().view(-1, 1)
		modules = [module for module in model_with_grads.modules() if len([param for param in module._parameters.values() if param is not None and param.requires_grad]) > 0]
		for module_index, module in enumerate(modules):
			optimizer = self.optimizers[module_index]
			if learning_module_index == module_index:
				optimizer(module, loss_t)
			else:
				with torch.no_grad():
					optimizer(module, loss_t)
		self.local_model.copy_params_to(model_with_grads)
		return self.local_model.model