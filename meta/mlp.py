from torch import nn


class MLP(nn.Module):

	def __init__(self, input_size, output_size, hidden_size):
		super(MLP, self).__init__()
		self.hidden_layer = nn.Linear(input_size, hidden_size, bias=True)
		self.hidden_activation = nn.ReLU()
		self.output_layer = nn.Linear(hidden_size, output_size, bias=True)

	def forward(self, x):
		h = self.hidden_activation(self.hidden_layer(x))
		return self.output_layer(h)