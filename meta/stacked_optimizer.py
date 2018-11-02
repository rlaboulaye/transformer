from torch import nn

from .local_model import LocalModel
from .lstm_optimizer import LSTMOptimizer


class StackedOptimizer(nn.Module):

	def __init__(self, model, hidden_size=20, num_layers=1, momentum=0):
		super(StackedOptimizer, self).__init__()
		self.local_model = LocalModel(model)
		self.initialize_optimizers(hidden_size, num_layers, momentum)

	def to(self, device):
		super(StackedOptimizer, self).to(device)
		for optimizer in self.optimizers:
			optimizer.to(device)

	def initialize_optimizers(self, hidden_size, num_layers, momentum):
		self.optimizers = nn.ModuleList()
		modules = [module for module in self.local_model.model.modules() if len([param for param in module._parameters.values() if param.requires_grad]) > 0]
		for module in modules:
			self.optimizers.append(LSTMOptimizer(module, hidden_size, num_layers, momentum))

	def reset_state(self, learn_initialization=True):
		for optimizer in self.optimizers:
			optimizer.reset_state(learn_initialization)

	def initialize_params(self, model, learn_initialization=True):
		if learn_initialization:
			for optimizer in self.optimizers:
				optimizer.initialize_params()
			self.local_model.copy_params_to(model)
		else:
			self.local_model.copy_params_from(model)

	def forward(self, model_with_grads, loss):
		loss_t = loss.clone().detach().view(-1, 1)
		modules = [module for module in model_with_grads.modules() if len([param for param in module._parameters.values() if param.requires_grad]) > 0]
		for module_index, module in enumerate(modules):
			optimizer = self.optimizers[module_index]
			optimizer(module, loss_t)
		self.local_model.copy_params_to(model_with_grads)
		return self.local_model.model