"""
Simple neural network has one hidden layer of size 4--predecessor to generalized NN in neuralnetwork.py
Made with guidance from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4) # Uniform random weights over [0, 1)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    # no biases implemented in simple neural network
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1)) # activation(input vector * weight vector) for each layer1 node
        self.output = sigmoid(np.dot(self.layer1, self.weights2)) # activation(layer1 vector * weight vector) for each output node
        return self.output

    # simple backprop: calculate gradients of cost with respect to weights1/2 and add it to current weights 1/2
    # cost function: (output - desired_output)^2
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, x, y):
        self.output = self.feedforward()
        self.backprop()

# Helper functions
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def sigmoid_derivative(p):
    return p * (1 - p)

# Simple NN example
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

NN = NeuralNetwork(X, y)
print("Input : \n" + str(X))
print("Actual Output: \n" + str(y))
for i in range(5000): # trains NN 5,000 times
    if i % 1000 == 0:
        print("iteration # " + str(i) + "\n")
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")
    NN.train(X, y)