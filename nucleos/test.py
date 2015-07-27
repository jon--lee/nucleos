import nucleos
import layers
import activations
import numpy as np
import colors

def output(s=""):
    print colors.green + str(s) + colors.end


net = nucleos.Network([2, 3, 1])
output(net.forward(np.array([[1],[1]])))
