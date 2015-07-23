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
        
        
    # propagate forward while recording vectors for optimization
    # given activations of previous layer
    def forward(self):    
        pass

    # propagate backward while using vectors from forward
    # propagation given mu vector of next layer
    def backward(self, mu):
        pass
