# General code for fully connected neural network.
import numpy
import math
import random

# set up initial values for 2x2 example
layers = [2,2]
weights = [[[0.9,0.2],[0.3,0.8]]]
inp = [[1], [0.5]]


# setting up random weights for structure of your layer
def set_up_weights(layers):
    weights = []
    for i in range(len(layers)-1):
        layer_weights = []
        for j in range(layers[i]):
            line = []
            for k in range(layers[i+1]):
                line.append(random.random())
            layer_weights.append(line)
        print(layer_weights)
        weights.append(layer_weights)
    return weights


def sigmoid(x):
    return 1/(1+math.e**-x)


def get_output(inp, weights):
    layer = inp
    for layer_w in weights:
        # perform matrix multiplication
        layer_mat = numpy.dot(layer_w, layer)

        # sum all the layers in matrix. Then apply sigmoid to each one of them.
        layer = [sigmoid(sum(layer_mat[i])) for i in range(len(layer_mat))]

    # return the values of the last layer (output layer)
    return layer


print(get_output(inp, weights))
