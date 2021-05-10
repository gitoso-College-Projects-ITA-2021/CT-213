import numpy as np
from utils import sigmoid, sigmoid_derivative


class NeuralNetwork:
    """
    Represents a two-layers Neural Network (NN) for multi-class classification.
    The sigmoid activation function is used for all neurons.
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs, alpha):
        """
        Constructs a three-layers Neural Network.

        :param num_inputs: number of inputs of the NN.
        :type num_inputs: int.
        :param num_hiddens: number of neurons in the hidden layer.
        :type num_hiddens: int.
        :param num_outputs: number of outputs of the NN.
        :type num_outputs: int.
        :param alpha: learning rate.
        :type alpha: float.
        """
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.alpha = alpha
        self.weights = [None] * 3
        self.biases = [None] * 3
        self.weights[1] = 0.001 * np.random.randn(num_hiddens, num_inputs)
        self.weights[2] = 0.001 * np.random.randn(num_outputs, num_hiddens)
        self.biases[1] = np.zeros((num_hiddens, 1))
        self.biases[2] = np.zeros((num_outputs, 1))

    def forward_propagation(self, inputs):
        """
        Executes forward propagation.
        Notice that the z and a of the first layer (l = 0) are equal to the NN's input.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :return z: values computed by applying weights and biases at each layer of the NN.
        :rtype z: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        :return a: activations computed by applying the activation function to z at each layer.
        :rtype a: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        """
        z = [None] * 3
        a = [None] * 3
        z[0] = inputs
        a[0] = inputs

        # Forward propagation
        for l in range(1, 3):
            # Update Z
            z[l] = self.weights[l] @ a[l-1] + self.biases[l]

            # Update A
            a[l] = sigmoid(z[l])

        return z, a

    def compute_cost(self, inputs, expected_outputs):
        """
        Computes the logistic regression cost of this network.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: list of numpy matrices.
        :return: logistic regression cost.
        :rtype: float.
        """
        z, a = self.forward_propagation(inputs)
        y = expected_outputs
        y_hat = a[-1]
        cost = np.mean(-(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))
        return cost

    def compute_gradient_back_propagation(self, inputs, expected_outputs):
        """
        Computes the gradient with respect to the NN's parameters using back propagation.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (num_outputs, num_samples) numpy array.
        :return weights_gradient: gradients of the weights at each layer.
        :rtype weights_gradient: 3-dimensional list of numpy arrays.
        :return biases_gradient: gradients of the biases at each layer.
        :rtype biases_gradient: 3-dimensional list of numpy arrays.
        """
        weights_gradient = [None] * 3
        biases_gradient = [None] * 3
        delta = [None] * 3

        z, a = self.forward_propagation(inputs)
        y = expected_outputs
        y_hat = a[-1]

        # Output layer delta
        delta[2] = np.array(y_hat - y)

        # Hidden layer delta
        delta[1] = (self.weights[2].T @ delta[2]) * sigmoid_derivative(z[1])

        # Biases Gradient
        biases_gradient[1] = np.mean(delta[1], axis=1)
        biases_gradient[1] = biases_gradient[1].reshape((biases_gradient[1].shape[0], 1))
        biases_gradient[2] = np.mean(delta[2], axis=1)
        biases_gradient[2] = biases_gradient[2].reshape((biases_gradient[2].shape[0], 1))

        # Weight Gradient
        weights_gradient[1] = []
        for k in range(len(delta[1])):
            aux = []
            for j in range(len(a[0])):
                aux.append(np.mean(delta[1][k] * a[0][j]))
            weights_gradient[1].append(aux)
        weights_gradient[1] = np.array(weights_gradient[1])

        weights_gradient[2] = []
        for c in range(len(delta[2])):
            aux = []
            for k in range(len(a[1])):
                aux.append(np.mean(delta[2][c] * a[1][k]))
            weights_gradient[2].append(aux)
        weights_gradient[2] = np.array(weights_gradient[2])

        return weights_gradient, biases_gradient

    def back_propagation(self, inputs, expected_outputs):
        """
        Executes the back propagation algorithm to update the NN's parameters.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (num_outputs, num_samples) numpy array.
        """
        weights_gradient, biases_gradient = self.compute_gradient_back_propagation(inputs, expected_outputs)
        
        # Add logic to update the weights and biases
        for i in range(1, 3):
            self.weights[i] = self.weights[i] - self.alpha * weights_gradient[i]
            self.biases[i] = self.biases[i] - self.alpha * biases_gradient[i]