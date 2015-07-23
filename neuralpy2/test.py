import layers
import activations
import numpy as np
import colors

def output(s=""):
    print colors.green + str(s) + colors.end


l = layers.layer(3, activations.sigmoid)
l.w = np.random.rand(3, 2)
l.b = np.random.rand(3, 1)

inputs = [1, 0]
inputs = np.array(inputs)
inputs = inputs.reshape((len(inputs), 1))

output(l.forward(inputs))
