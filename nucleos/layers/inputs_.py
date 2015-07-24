# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# Implementation of layer interface for
# input layer. retains super intitializer.
# no weight or bias caculations
# no backward propagation
# only append, no prepend (assuming this is the first layer)
# 

# system libraries
# internal libraries
import layers
# third party libraries
import numpy as np

class input_(layers.layer):
    
    def __init__(self, size):
        super(input_, self).__init__(size, None)
        self.type_ = layers.type_input

    # Implementation of layer interface
    # for input vector layer, no calculations
    def forward(self, x):
        self.x = x
        return x

    # assign to instance var next_
    # prepend self to next_
    # caller beware: may potentially overwrite next_'s weights
    def append(self, next_):
        self.next_ = next_
        next_.prepend(self)
