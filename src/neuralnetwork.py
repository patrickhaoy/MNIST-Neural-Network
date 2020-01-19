"""
Generalized simple neural network with any number of layers and neurons. Uses mini-batch stochastic gradient
descent to train, cross-entropy cost function, and sigmoid activation function.
Made with guidance from http://neuralnetworksanddeeplearning.com/chap1.html
"""

import numpy as np
import random


class NeuralNetwork:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes # sizes is a vector describing number of neurons in each layer
        # Biases and weights generated randomly as X ~ N(0, 1)
        # each element of the vectors represent a weight/bias matrix for a different layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Trains the neural network using mini-batch stochastic gradient descent
        :param training_data: list of tuples (x, y) representing (training input, desired output)
        :param test_data: if provided, network will test against test data after each epoch
        """
        # save lengths of data to speed performance
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.training_step(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def training_step(self, mini_batch, eta):
        """
        Calculates cost function gradients, modifies weights and biases based on gradient
        """
        # Create multi-D array to store biases/weights
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y) # Calculate gradient vector
            # Add together gradient at each training step
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

        # Approximate gradient: average mini-batch gradients by dividing over len(mini_batch)
        # eta or training rate slows speed of training
        self.weights = [w - (eta / len(mini_batch)) * gw
                        for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - (eta / len(mini_batch)) * gb
                       for b, gb in zip(self.biases, grad_b)]

    def backprop(self, x, y):
        """
        :param x: training input
        :param y: desired output
        :return: gradient of weights and biases for single training data
        """
        # Activation of input layer is itself
        activation = x
        activations = [x]

        # keep track of all the zs (i.e. weight*previous_layer_neuron_activation_value+bias)
        zs = []

        # Feed-forward to get all the zs and activation values
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Initialize storage of gradients
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # Get delta for last layer: cost_derivative * sigma_prime(z(L))
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # Calculate gradient of last layer and store
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose()) # a(L-1) * delta

        # Repeat backpropation for rest of layers going backwards
        # Each element of grad_b/w is a gradient vector for the weights/biases of a different layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].T)
        return grad_b, grad_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


# Helper functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Cross-entropy cost_derivative
def cost_derivative(output_activations, y):
    return output_activations - y
