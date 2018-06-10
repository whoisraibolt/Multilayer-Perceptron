#!/usr/bin/pdthon
# -*- coding: utf-8 -*-

# Imports
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# Dataset definition - all combinations of XOR
x = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
]

# Desired output
d = [
	[0],
	[1],
	[1],
	[0]
]

# Set Parameters

# Learning rate - [0, 1]
learning_rate = 0.05

# Maximum number allowed of iterations
max_iterations = 100000

# Number of epochs
n_epochs = 10000

# Total amount of training items
n_training = len(x)

# Set Network Parameters

# Input layer
n_input_nodes = 2

# Hidden layer 1
n_hidden_nodes_1 = 1

# Output layer
n_output_nodes = 1

# Reserved spaces for storing input and output data
x_ = tf.placeholder(tf.float32, [None, n_input_nodes])
d_ = tf.placeholder(tf.float32, [n_training, n_output_nodes])

# Configure the parameters for the network

# Hidden layer 1

# Weights
theta1 = tf.Variable(tf.random_uniform([n_input_nodes,n_hidden_nodes_1], -1, 1))

# Bias
bias1 = tf.Variable(tf.zeros([n_hidden_nodes_1]))

# Net1 - Multiply the matrices
net1 = tf.matmul(x_, theta1)

# Activation function
O1 = tf.sigmoid(net1 + bias1)

# Output layer

# Hidden node weights of index 4  - [wi4]
theta2_x = tf.Variable(tf.random_uniform([n_input_nodes,n_output_nodes], -1, 1))
theta2_h = tf.Variable(tf.random_uniform([n_hidden_nodes_1,n_output_nodes], -1, 1))

# Hidden node bias of index 4
bias2 = tf.Variable(tf.zeros([n_output_nodes]))

# Net4 - Multiply the matrices
net4_x = tf.matmul(x_, theta2_x)
net4_h = tf.matmul(O1, theta2_h)

# Hidden node activation function of index 4
y = tf.sigmoid((net4_x + net4_h) + bias2)

# Cost (error) function and optimizer

# Mean Squared Estimate (MSE)
cost = tf.reduce_mean(tf.square(d_ - y))

# Training
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Create session
session = tf.Session()

# Initialize variables
init = tf.global_variables_initializer()

# Run the session
session.run(init)

# Training Step

# Start-time used for printing time-usage below
start_time = time.time()

for i in range(max_iterations):
	session.run(train_step, feed_dict={x_: x, d_: d}) # Dictionary object with placeholders as keys and feed tensors as values

	if i % n_epochs == 0:
		accuracy = session.run(cost, feed_dict={x_: x, d_: d})

		print("Optimization Iteration: %s, Training Accuracy: %s" % (i, accuracy))

# Ending time.
end_time = time.time()

# Difference between start and end-times.
time_dif = end_time - start_time

# Print the time-usage.
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Test Step

# Returns the absolute value, if the value is less than 0.5
correct_prediction = abs(d_ - y)

# Calculates the mean of the elements
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

final_accuracy = session.run(accuracy, feed_dict={x_: x, d_: d})
print'\nAccuracy on Test Step: %s',final_accuracy