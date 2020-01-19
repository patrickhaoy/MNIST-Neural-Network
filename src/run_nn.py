"""
Runs MNIST data as training and test data for neuralnetwork.py
"""

import mnist_loader
import neuralnetwork


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
nn = neuralnetwork.NeuralNetwork([784, 30, 10])
# Use MNIST training_data to train NN over 30 epochs, with mini-batch size of 10, and a learning rate of 3.0
nn.train(training_data, 30, 10, 3.0, test_data=test_data)