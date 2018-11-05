import copy


class LocalModel(object):

	def __init__(self, model):
		self.model = copy.deepcopy(model)

	def copy_params_from(self, model):
		for local_params, params in zip(self.model.parameters(), model.parameters()):
			local_params.data.copy_(params.data)

	def copy_params_to(self, model):
		for local_params, params in zip(self.model.parameters(), model.parameters()):
			params.data.copy_(local_params.data)