# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# Implementation of layer interface for
# fully connected feedforward multilayer perceptron
# retains super initializer
# weights not initialized by default but biases are as random
# append should be only interface for joining adjacent layers
# do NOT use prepend, append will prepend for you.
# prepend assigns random and appropriately-sized weights
# 

# system libraries
# internal libraries
import layer
# third party libraries
import numpy as np


class MLP(layer.Layer):

    # initialize mlp with type (as with all layers)
    # and initialize the random biases
    def __init__(self, size, activ):
        super(MLP, self).__init__(size, activ)
        self.type_ = layer.type_mlp
        self.b = np.random.rand(size, 1)
        self.delta_b = np.zeros(self.b.shape)
        
    # Implementation of layer interface for
    # fully connected feedforward multilayer perceptron
    def forward(self, incoming_activations):
        self.a = incoming_activations
        self.z = np.dot(self.w, self.a) + self.b
        self.x = self.activ.func(self.z)
        return self.x

    def backward(self, mu):
        self.delta_b += mu
        self.delta_w += np.dot(mu, self.a.transpose())
        if self.prev.z is not None:
            mu = np.dot(self.w.transpose(), mu) * self.activ.deriv(self.prev.z)
            return mu
        else:
            return None


    # set next_ layer to instance var.
    # prepend self to next_.
    # caller beware: will potentially overwrite next_'s weights
    def append(self, next_):
        self.next_ = next_
        next_.prepend(self)

    # set prev layer to instance var.
    # initialize the weights as random between the layers
    # caller beware: will overwrite self's weights
    def prepend(self, prev):
        self.prev = prev
        self.w = np.random.rand(self.size, prev.size)
        self.zero_deltas()

    # randomize parameters for fully connected
    # which is just random weights for each connectin
    # with preceding layer and full vector of random biases
    def randomize_parameters(self):
        if self.prev is not None:
            self.w = np.random.rand(self.size, prev.size)
        self.b = np.random.rand(self.size, 1)


