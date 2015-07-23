# system libraries
# internal libraries
from layers import mlp
import activations
# third party libraries
import numpy as np

# class for the neural network. initialized with list of layer sizes
class Network(object):

    def __init__(self, sizes):
        
        self.sizes = sizes        
    


