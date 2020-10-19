import numpy as np
from math import log

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def dsigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    f = tanh(x)
    return 1 - (f*f)

def binary_cross_entropy(actual_output, label):
    return -(label * log(actual_output) + (1 - label) * log(1 - actual_output))

def l2_loss(actual_output, label):
    return (label - actual_output)^2

def d_l2_loss(actual_output, label):
    return 2 * (label - actual_output)