#
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# Network module that defines a network. sizes parameter determines
# the number of nodes in each respective layer. layers by default
# consist of an input_ and mlps with sigmoid activations.
# 


# system libraries
# internal libraries
from layers import mlp, input_
import activations
# third party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):

        it = iter(size)
        self.root = input_(next(it))              
        
        for size in it:
            next_layer = mlp(size, activations.sigmoid)
            self.root.append(next_layer)

    def forward(x):
        for layer in layers:
            x = layer.forward(x)
        return x
            
    def append(layer):
        self.layers.append(layer)



