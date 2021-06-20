import numpy as np
import random

# Importing the MNIST digits data set
# from keras.datasets import mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


class DigitsNeuralNetwork(object):

    def __init__(self, sizes):
        self.sizes = sizes  # list of numbers, each number represents a layer's size
        self.num_layers = len(sizes)  # number of layers in network

        # Random weights and biases, from the standard normal distribution.
        self.biases = []
        self.weights = []
        for y in sizes[1:]:
            self.biases = np.random.randn(y, 1)
        for x, y in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(y, x))

    # Calculate and return the last layer's neurons' values.
    # input_layer is a vector containing the input layer's neurons' values.
    def output_layer(self, input_layer):
        layer = input_layer
        for b, w in zip(self.biases, self.weights):
            layer = sigmoid(np.dot(w, layer)+b)
        return layer

    # Gradient descent. training_data is a list of (x, y) pairs, where the
    # first element is a training input and the second is the expected output.
    def gradient_descent(self, training_data, rounds, mini_batch_size, learning_rate, test_data):

        n_test = len(test_data)
        n = len(training_data)
        for j in range(rounds):
            random.shuffle(training_data)

            mini_batches = []
            for k in range(0, n, mini_batch_size):
                mini_batches.append(training_data[k:k + mini_batch_size])

            # Train using every mini batch
            for mini_batch in mini_batches:
                # gradient descent formula (with delta and upside down delta)
                nabla_b = []
                nabla_w = []
                for b in self.biases:
                    nabla_b.append(np.zeros(b.shape))
                for w in self.weights:
                    nabla_w.append(np.zeros(w.shape))

                # back propagation
                for x, y in mini_batch:
                    delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                # update weights and biases
                self.weights = [w - (learning_rate / len(mini_batch)) * nw
                                for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (learning_rate / len(mini_batch)) * nb
                               for b, nb in zip(self.biases, nabla_b)]

            # Test the network on the test dataset, then print success rate
            successfulPredictions = self.evaluate(test_data)
            print("Round number " + str(j) + ": " + str(successfulPredictions / n_test))

    def back_propagation(self, x, y):

        nabla_b = []
        nabla_w = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))

        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    # Return number of correct predictions of test data
    def evaluate(self, test_data):
        test_results = []
        for (x, y) in test_data:
            # 'argmax' returns index of maximum element
            test_results.append((np.argmax(self.output_layer(x)), y))

        correctPredictions = 0
        for (x, y) in test_results:
            if x == y:
                correctPredictions += 1
        return correctPredictions

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def recognize(self, x):
        layer = x
        # list of all layers
        layers = [x]
        # list to store all the w*layer dot-multiplication vectors, before using sigmoid
        mult = []
        for b, w in zip(self.biases, self.weights):
            vec = np.dot(w, layer) + b  # dot multiplication
            mult.append(vec)  # add
            layer = sigmoid(vec)
            layers.append(layer)
        return layers[-1]

