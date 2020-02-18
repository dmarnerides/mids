#!/usr/bin/env python
# Author: Demetris Marnerides

import torch
from torch import nn
import time
from torch.nn import Parameter
import util

######################################################################
# Data
######################################################################
NUM_POINTS = 40
NUM_TRAIN = int(NUM_POINTS * 0.8)
#  x, y = util.get_data('line', NUM_POINTS, noise=0.02)
x, y = util.get_data('sine2', NUM_POINTS, noise=0)
# Show the data
#  util.plot(x, y, 'training')

# Split in training and validation
x_train, y_train = x[:NUM_TRAIN], y[:NUM_TRAIN]
x_valid, y_valid = x[NUM_TRAIN:], y[NUM_TRAIN:]

# Show training and validation
util.plot(x_train, y_train, 'training')
util.plot(x_valid, y_valid, 'validation')


######################################################################
# Model
######################################################################


#  class LineNN(nn.Module):
#      def __init__(self):
#          super(LineNN, self).__init__()
#          self.line = nn.Linear(1, 1)
#          self.line.weight.data.fill_(2)
#          self.line.bias.data.fill_(0)
#
#      def forward(self, x):
#          return self.line(x)


class MultilayerNN(nn.Module):
    def __init__(self, num_layers=1, num_hidden=1, activation=nn.ReLU):
        super(MultilayerNN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.layers.append(nn.Linear(1, num_hidden))
        self.bn.append(nn.BatchNorm1d(num_hidden))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.bn.append(nn.BatchNorm1d(num_hidden))
        self.layers.append(nn.Linear(num_hidden, 1))
        self.activation = activation()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


#  my_f = LineNN()
my_f = MultilayerNN(num_layers=1, num_hidden=500)

#  util.plot_function(my_f)
#  util.wait_and_exit()

######################################################################
# Fit to data
######################################################################


def my_loss(prediction, target):
    return torch.abs((prediction - target))


def evaluate_loss_for_training_data():
    total_loss = 0
    batch_size = 8
    num_batches = int(NUM_TRAIN // batch_size)
    for i in range(num_batches):
        fro = i * batch_size
        to = min((i + 1) * batch_size, len(x_train))
        current_prediction = my_f(x_train[fro:to].unsqueeze(-1))
        current_target = y_train[fro:to].unsqueeze(-1)
        total_loss += my_loss(current_prediction, current_target)
    return total_loss.mean()


#  loss = evaluate_loss_for_training_data()
#  print(f'Loss: {loss.item():.2f}')

#  optimizer = torch.optim.SGD(my_f.parameters(), lr=0.01)
optimizer = torch.optim.Adam(my_f.parameters())


def single_step_gradient_descent():
    loss = evaluate_loss_for_training_data()  # EVALUATE LOSS
    loss.backward()  # EVALUATE GRADIENT (BACKPROPAGATION)

    # Move the parameters in the direction of the gradient
    optimizer.step()

    # Print some results
    print(f'loss = {loss.item():.2f}')
    optimizer.zero_grad()


#  single_step_gradient_descent()
#  util.plot_function(my_f)
#  util.wait_and_exit()


util.set_pause_time(0.001)
for i in range(1000):
    single_step_gradient_descent()
    util.plot_function(my_f, keep=False)


util.wait_and_exit()
