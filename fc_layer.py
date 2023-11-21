from layer import Layer
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        # Randomize the weights initially
        # weights have shape (input, output) or i x j
        self.weights = np.random.rand(input_size, output_size) - 0.5
        # bias has shape 1 x j
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        # (1 x i) * (i x j) + (1 x j) = (1 x j)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        # 1. dE/dX = dE/dY * W.T
        #   (1 x i) = (1 x j) * (j x i)
        input_error = np.dot(output_error, self.weights.T)
        # 2. dE/dW = X.T * dE/dY 
        #   (i x j) = (i x 1) * (1 x j)
        weights_error = np.dot(self.input.T, output_error)
        # 3. dE/dB = dE/dY
        bias_error = output_error

        # update weights and bias term
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error