#!/usr/bin/env python
# Author: Demetris Marnerides

import torch
from torch.nn import Parameter
import util

######################################################################
# Data
######################################################################
NUM_POINTS = 40
NUM_TRAIN = int(NUM_POINTS * 0.8)
x, y = util.get_data('sine', NUM_POINTS, noise=0.02)
# Show the data
#  util.plot(x, y, 'training')

# Split in training and validation
x_train, y_train = x[:NUM_TRAIN], y[:NUM_TRAIN]
x_valid, y_valid = x[NUM_TRAIN:], y[NUM_TRAIN:]

# Show training and validation
util.plot(x_train, y_train, 'training')
util.plot(x_valid, y_valid, 'validation')

#  util.wait_and_exit()

######################################################################
# Model
######################################################################
a = Parameter(torch.Tensor([2]))
b = Parameter(torch.Tensor([0]))

#  print(a)
#  print(b)


def my_f(x):
    result = a * x + b
    return result


util.plot_function(my_f)

#  util.wait_and_exit()

######################################################################
# Fit to data
######################################################################


def my_loss(prediction, target):
    return (prediction - target) ** 2


def evaluate_loss_for_training_data():
    total_loss = 0
    for i in range(NUM_TRAIN):
        #  current_prediction = my_f(x_train[i])
        current_target = y_train[i]
        total_loss = total_loss + (current_target - my_f(x_train[i])) ** 2
        #  total_loss += my_loss(current_prediction, current_target)
    return total_loss


#  loss = evaluate_loss_for_training_data()
#  print(f'Loss: {loss.item():.2f}')

#  util.wait_and_exit()


def single_step_gradient_descent():
    LR = 0.02
    loss = evaluate_loss_for_training_data()  # EVALUATE LOSS
    loss.backward()  # EVALUATE GRADIENT (BACKPROPAGATION)

    # Move the parameters in the direction of the gradient
    a.data.add_(-LR * a.grad)
    b.data.add_(-LR * b.grad)

    # Print some results
    print(
        f'> Step: a_grad={a.grad.item():.2f}, '
        f'b_grad={b.grad.item():.2f}, '
        f'loss = {loss.item():.2f}'
    )
    a.grad.zero_()
    b.grad.zero_()


#  single_step_gradient_descent()
#  util.plot_function(my_f, keep=False)
#  util.wait_and_exit()


util.set_pause_time(0.05)
for i in range(100):
    single_step_gradient_descent()
    util.plot_function(my_f, keep=False)


util.wait_and_exit()
