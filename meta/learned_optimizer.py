from torch import nn

from lstm_update_rule import LSTMUpdateRule


class LearnedOptimizer(nn.Module):

	def __init__(self, params):
		super(LearnedOptimizer, self).__init__()
		self.update_rule = LSTMUpdateRule(params)

	def initialize_params(self):
		return self.update_rule.initialize_params()

	def set_params(self, params):
		self.update_rule.set_params(params)

	def step(self, closure=None):

		loss = None
		if closure is not None:
			loss = closure()

		for group_index, group in enumerate(self.param_groups):
			for p_index, p in enumerate(group['params']):
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError('MetaOptimizer does not support sparse gradients')

				# figure out where to get loss from
				p.data = self.update_rule(p.data, grad, 0, group_index, p_index)

		return loss