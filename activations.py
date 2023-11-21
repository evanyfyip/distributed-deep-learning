import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    # Note this is leaky relu
    greater_than_0 = x * (x >= 0)
    less_than_0 = 0.01 * x * (x < 0)
    return greater_than_0 + less_than_0

def relu_prime(x):
    greater_than_0 = 1 * (x >= 0)
    less_than_0 = 0.01 * (x < 0)
    return greater_than_0 + less_than_0