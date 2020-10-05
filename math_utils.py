import numpy as np
from math import log

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x)
    return 1 - (x*x)

def mean_absolute_error(actual_output, label):
    return abs(actual_output - label)

def binary_cross_entropy(actual_output, label):
    return -(label * log(actual_output) + (1 - label) * log(1 - actual_output))

