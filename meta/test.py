import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.nn.modules.loss import MSELoss

from stacked_optimizer import StackedOptimizer
from mlp import MLP


def train_network(train_set, test_set, loss_function, optimizer, meta_optimizer, device, epochs):

	mlp = MLP(X_Y.shape[-1] - 1, 1, 32)
	mlp.to(device)

	optimizer.reset_state()
	optimizer.initialize_params(mlp)

	#
	# optimizer = Adam(mlp.parameters(), lr=.001)
	# optimizer = SGD(mlp.parameters(), lr=.000003)
	#

	for epoch in range(epochs):
		##
		loss_sum = 0
		previous_loss = torch.zeros(1, device=device)
		##
		losses = []
		for batch in train_set:
			x = batch[:,:-1]
			y = batch[:,-1]
			x = torch.tensor(x, dtype=torch.float32, device=device)
			y = torch.tensor(y, dtype=torch.float32, device=device)
			y_hat = mlp(x).view(-1)
			loss = loss_function(y_hat, y).sqrt()
			mlp.zero_grad()
			loss.backward()
			# optimizer.step()
			tuned_mlp = optimizer(mlp, loss)
			losses.append(loss)
			##
			# y_hat = tuned_mlp(x).view(-1)
			# loss = loss_function(y_hat, y).sqrt()
			# loss_sum += (loss - previous_loss.detach())
			# previous_loss = loss.data
			##
		losses = torch.cat([loss.unsqueeze(-1) for loss in losses], dim=-1)
		loss = losses.mean(-1)
		print('Epoch {} Train Loss: {}'.format(epoch, loss.cpu().item()))
		##
		# meta_optimizer.zero_grad()
		# loss_sum.backward(retain_graph=True)
		# meta_optimizer.step()
		##
	mlp = tuned_mlp
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
	meta_optimizer.zero_grad()
	loss.backward()
	# print('Grad: {}'.format(optimizer.optimizers[0].W_theta.weight.grad))
	# print('Grad: {}'.format(optimizer.optimizers[0].W_grad.weight.grad))
	# print('Grad: {}'.format(optimizer.optimizers[1].W_theta.weight.grad))
	# print('Grad: {}'.format(optimizer.optimizers[1].W_grad.weight.grad))
	meta_optimizer.step()
	print('Epoch Test Loss: {}'.format(loss.cpu().item()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta_epochs = 100
epochs = 20

batch_size = 8
train_test_split = .08

df = pd.read_csv('data.csv')
X_Y = df.values
X_Y = X_Y.reshape(-1, batch_size, X_Y.shape[-1])
train_set = X_Y[:round(train_test_split * X_Y.shape[0])]
test_set = X_Y[round(train_test_split * X_Y.shape[0]):]

loss_function = MSELoss()

mlp = MLP(X_Y.shape[-1] - 1, 1, 32)

optimizer = StackedOptimizer(mlp)
optimizer.to(device)
meta_optimizer = Adam(optimizer.parameters(), lr=.001)

for meta_epoch in range(meta_epochs):
	print('Meta Epoch {}'.format(meta_epoch))
	X_Y = np.random.permutation(X_Y)
	train_set = X_Y[:round(train_test_split * X_Y.shape[0])]
	test_set = X_Y[round(train_test_split * X_Y.shape[0]):]
	train_network(train_set, test_set, loss_function, optimizer, meta_optimizer, device, epochs)
