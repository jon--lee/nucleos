from layers import mlp
import numpy as np
import activations
l = mlp(3, activations.sigmoid)

l.w = np.random.rand(3,2)
l.b = np.random.rand(3, 1)
print l.forward([[1],[1]])
