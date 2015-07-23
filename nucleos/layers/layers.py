# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# layer defines a interface layer and related functions such
# as forward and back propagation.
# 
# parameters are typically:
#   size - integer of size of layer (number of activations)
#   act - activation function as defined by activations module
# so the initializer is actually an implementation for declaration of
# instance variables
# weights and biases not set by default
#


# system libraries
# internal libraries
# third party libraries
import numpy as np

class layer():

    
    def __init__(self, size, activ):
        self.size = size    # num nodes in layer
        self.activ = activ  # activation function
        
        self.w = None       # incoming weights  (size x prev-size) matrix
        self.b = None       # incoming biases   (size x 1) vector
        
        self.a = None       # incoming activation vector from previous layer
        self.z = None       # weighted sum + bias vector
        self.x = None       # activation function on z vector

        self.next_ = None   # layer that comes after this
        self.prev = None    # layer that comes before this
        
        self.type_ = type_gen
        
    # propagate forward while recording vectors for optimization
    # given activations of previous layer
    def forward(self, incoming_activations):    
        raise NotImplementedError

    # propagate backward while using vectors from forward
    # propagation given mu vector of next layer
    def backward(self, mu):
        raise NotImplementedError
    
    # use only this method to add a layer to the list
    # append function should be the default interface
    def append(self, next_):
        raise NotImplementedError

    # Do not call this method from a non-layer caller.
    # prepend function that should also establish
    # the weight connection between the adjacent layers.
    def prepend(self, prev):
        raise NotImplementedError



type_gen = "gen"
type_input = "in"
type_mlp = "mlp"
