class LocalModel(object):

	def __init__(self, model):
		self.model = model

	# def reset(self):
	# 	for group in self.model.parameters():
	# 		group.data = group.detach().data

	# def get_params(self):
	# 	return list(self.model.parameters())

	# def set_params(self, params):
	# 	for group_index, group in enumerate(self.model.parameters()):
	# 		group.data.copy_(params[group_index].data)

	# def copy_params_from(self, model):
	# 	for model_local, model in zip(self.model.parameters(), model.parameters()):
	# 		model_local.data.copy_(model.data)

	def copy_params_to(self, model):
		for local_params, params in zip(self.model.parameters(), model.parameters()):
			params.data.copy_(local_params.data)