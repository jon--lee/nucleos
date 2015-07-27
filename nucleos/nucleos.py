#
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# Network module that defines a network. sizes parameter determines
# the number of nodes in each respective layer. layers by default
# consist of an inputs and mlps with sigmoid activations.
# 


# system libraries
# internal libraries
from layers import MLP, Input
import activations
import colors
# third party libraries
import numpy as np

def output(s=""):
    print colors.green + str(s) + colors.end

# ITERATOR CODE UNTESTED, TEST LATER


class Network(object):

    def __init__(self, sizes):

        it = iter(sizes)
        self.start = Input(next(it))              
        layer = self.start
        for size in it:
            layer.append( MLP(size, activations.sigmoid) )
            layer = layer.next_
        self.end = layer

    def forward(self, x):
        it = iter(self.start)
        for layer in it:
            x = layer.forward(x)
        return x

    def append(self, layer):
        self.end.append(layer)
        self.end = self.end.next_



