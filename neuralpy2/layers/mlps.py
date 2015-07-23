# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# Implementation of layer interface for
# fully connected feedforward multilayer perceptron
# retains super initializer
# 

# system libraries
# internal libraries
import layers
# third party libraries
import numpy as np


class mlp(layers.layer):
    
    # Implementation of layer interface for
    # fully connected feedforward multilayer perceptron
    def forward(self, incoming_activations):
        self.a = incoming_activations
        self.z = np.dot(self.w, self.a) + self.b
        self.x = self.activ.func(self.z)
        return self.x


    def backward(self, mu):
        return mu
    
        
