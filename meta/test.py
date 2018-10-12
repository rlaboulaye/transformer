import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.modules.loss import MSELoss

from lstm_optimizer import LSTMOptimizer


class MLP(nn.Module):

	def __init__(self, input_size, output_size, hidden_size):
		super(MLP, self).__init__()
		self.hidden_layer = nn.Linear(input_size, hidden_size, bias=True)
		self.hidden_activation = nn.ReLU()
		self.output_layer = nn.Linear(hidden_size, output_size, bias=True)

	def forward(self, x):
		h = self.hidden_activation(self.hidden_layer(x))
		return self.output_layer(h)

def train_network(train_set, test_set, optimizer, device, epochs):

	epochs = 5

	mlp = MLP(X_Y.shape[-1] - 1, 1, 32)
	mlp.to(device)
	loss_function = MSELoss()

	optimizer.set_params(mlp.parameters())
	optimizer.initialize_params(mlp.parameters())

	for epoch in range(epochs):
		mlp.train()
		losses = []
		for batch in train_set:
			x = batch[:,:-1]
			y = batch[:,-1]
			x = torch.tensor(x, dtype=torch.float32, device=device)
			y = torch.tensor(y, dtype=torch.float32, device=device)
			y_hat = mlp(x).view(-1)
			loss = torch.sqrt(loss_function(y_hat, y))
			loss.backward()
			optimizer(loss)
			optimizer.zero_grad()
			losses.append(loss.cpu().item())
		print('Epoch {}: {}'.format(epoch, np.mean(losses)))
	with torch.no_grad():
		mlp.eval()
		losses = []
		for batch in test_set:
			x = batch[:,:-1]
			y = batch[:,-1]
			x = torch.tensor(x, dtype=torch.float32, device=device)
			y = torch.tensor(y, dtype=torch.float32, device=device)
			y_hat = mlp(x).view(-1)
			loss = torch.sqrt(loss_function(y_hat, y))
			losses.append(loss)
		losses = torch.cat([loss.unsqueeze(-1) for loss in losses], dim=-1)
		loss = losses.mean(-1)
		return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
train_test_split = .7

df = pd.read_csv('data.csv')
X_Y = df.values
X_Y = X_Y.reshape(-1, batch_size, X_Y.shape[-1])
train_set = X_Y[:round(train_test_split * X_Y.shape[0])]
test_set = X_Y[round(train_test_split * X_Y.shape[0]):]

mlp = MLP(X_Y.shape[-1] - 1, 1, 32)

# param_groups = list(mlp.parameters())
# for group in param_groups:
# 	print(group.shape)
# 	print(group)
# import sys
# sys.exit(1)

optimizer = LSTMOptimizer(mlp.parameters())
optimizer.to(device)
meta_optimizer = Adam(optimizer.parameters(), lr=.001)

meta_epochs = 1
epochs = 5
for meta_epoch in range(meta_epochs):
	print('Meta Epoch {}'.format(meta_epoch))
	loss = train_network(train_set, test_set, optimizer, device, epochs)
	loss.backward()
	meta_optimizer.step()
	meta_optimizer.zero_grad()
	print(loss.cpu().item())