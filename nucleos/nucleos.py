#
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# NetworkBase module that defines an abstract network. sizes parameter determines
# the number of nodes in each respective layer. layers by default
# consist of an inputs and mlps with sigmoid activations.
# 


# system libraries
import random
# internal libraries
from layers import MLP, Input
import activations
import colors
import costs
# third party libraries
import numpy as np

def output(s=""):
    print colors.green + str(s) + colors.end


class NetworkBase(object):

    def __init__(self, sizes):

        it = iter(sizes)
        self.start = Input(next(it))              
        layer = self.start
        for size in it:
            layer.append( MLP(size, activations.sigmoid) )
            layer = layer.next_
        self.end = layer

    # propagate forward by iterating over all layers
    # staring with the first layer
    def forward(self, x):
        it = iter(self.start)
        for layer in it:
            x = layer.forward(x)
        return x

    # append layer to the end layer of the network
    # simply by using the end instance variable
    def append(self, layer):
        self.end.append(layer)
        self.end = self.end.next_

    # pop the last layer off the network
    # by setting end instance var to the 
    # previous layer and removing the previous
    # layers forward propagation to this
    def pop(self):
        self.end = self.end.prev
        self.end.append(None)

    # train neural network based on training data
    # to optimize the weights. Abstract method
    # intended to be handled by implementing network
    # class
    def train(self, training_set, epochs, alpha):
        raise NotImplementedError

    # iterate through layer and update the biases
    # and weights as each layer type should handle
    # the individual implementation uniquely
    # applying the delta updates should always zero
    # out the delta weights
    def apply_updates(self, alpha):
        it = iter(self.start)
        for layer in it:
            layer.apply_updates(alpha)
            layer.zero_deltas()

    # iterate through each alyer and zero out
    # the weights, as handled by the implementations
    # of the given layers
    def zero_deltas(self):
        it = iter(self.start)
        for layer in it:
            layer.zero_deltas()


# Network class provides the implementation of NetworkBase's 
# non implemented functions.
class Network(NetworkBase):


    def train(self, training_set, epochs, alpha):

        alpha = float(alpha)
        n = len(training_set)

        self.zero_deltas()
        for j in xrange(epochs):
            #random.shuffle(training_set)
            self.update_batch(training_set, alpha)


    def update_batch(self, training_set, alpha):

        for x, y in training_set:

            self.forward(x)
            mu = costs.mean_square.deriv(self.end.x, y) * self.end.activ.deriv(self.end.z)
            root = self.end
            while root is not None:
                mu = root.backward(mu)
                root = root.prev


        self.apply_updates(alpha)

