# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# Implementation of layer interface for
# input layer. retains super intitializer.
# no weight or bias caculations
# no backward propagation
# 

# system libraries
# internal libraries
import layers
# third party libraries
import numpy as np

class input_(layers.layer):
    
    def __init__(self, size, activ):
        super(input_, self).__init__(size, activ)
        self.type_ = layers.type_input

    # Implementation of layer interface
    # for input vector layer
    def forward(self, x):
        self.x = x
        return x

    # assign to instance var next_
    # prepend self to next_
    # caller beware: may potentially overwrite next_'s weights
    def append(self, next_):
        self.next_ = next_
        next_.preprend(self)
