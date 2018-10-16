import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.modules.loss import MSELoss

from stacked_optimizer import StackedOptimizer
from mlp import MLP


def train_network(train_set, test_set, optimizer, meta_optimizer, device, epochs):

	epochs = 5

	mlp = MLP(X_Y.shape[-1] - 1, 1, 32)
	mlp.to(device)
	loss_function = MSELoss()

	optimizer.set_params(mlp.parameters())
	optimizer.initialize_params(mlp.parameters())
	# print(list(mlp.parameters())[0].data)
	for epoch in range(epochs):
		losses = []
		for batch_index, batch in enumerate(train_set):
			x = batch[:,:-1]
			y = batch[:,-1]
			x = torch.tensor(x, dtype=torch.float32, device=device)
			y = torch.tensor(y, dtype=torch.float32, device=device)
			y_hat = mlp(x).view(-1)
			loss = torch.sqrt(loss_function(y_hat, y))
			mlp.zero_grad()
			loss.backward()
			optimizer(loss)
			losses.append(loss)
		losses = torch.cat([loss.unsqueeze(-1) for loss in losses], dim=-1)
		loss = losses.mean(-1)
		print('Epoch {}: {}'.format(epoch, loss.cpu().item()))
	###
	theta_t = optimizer.theta_tm1[0]
	loss = theta_t.view(-1).abs().sum().sqrt()
	print('Grad: {}'.format(optimizer.W_theta.weight.grad))
	loss.backward()
	print('Grad: {}'.format(optimizer.W_theta.weight.grad))
	import sys
	sys.exit(0)
	###
	mlp = MLP(X_Y.shape[-1] - 1, 1, 32)
	optimizer.update_params(mlp.parameters())
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
	# print(optimizer.W_theta.weight)
	meta_optimizer.zero_grad()
	loss.backward()
	# for param in optimizer.parameters():
	# 	print(param.grad)
	print('Grad: {}'.format(optimizer.W_theta.weight.grad))
	meta_optimizer.step()
	# print(optimizer.W_theta.weight)
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

optimizer = StackedOptimizer(mlp)
optimizer.to(device)
meta_optimizer = Adam(optimizer.parameters(), lr=.001)

###
loss_function = MSELoss()
mlp = MLP(X_Y.shape[-1] - 1, 1, 32)
mlp.to(device)
optimizer.reset_state()
optimizer.initialize_params(mlp)

batch = train_set[0]
x = batch[:,:-1]
y = batch[:,-1]
x = torch.tensor(x, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.float32, device=device)
y_hat = mlp(x).view(-1)
loss = torch.sqrt(loss_function(y_hat, y))
mlp.zero_grad()
loss.backward()
local_mlp = optimizer(mlp, loss)

y_hat = local_mlp(x).view(-1)
loss = torch.sqrt(loss_function(y_hat, y))
meta_optimizer.zero_grad()
loss.backward()
print(optimizer.optimizers[0].W_theta.weight.grad)
meta_optimizer.step()
###

# meta_epochs = 5
# epochs = 5
# for meta_epoch in range(meta_epochs):
# 	print('Meta Epoch {}'.format(meta_epoch))
# 	loss = train_network(train_set, test_set, optimizer, meta_optimizer, device, epochs)


### haven't yet properly initialized the weights of either model