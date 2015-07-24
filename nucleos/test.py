from layers import *
import activations
import nucleos
import numpy as np





"""
inp = input_(2)
out = mlp(1, activations.sigmoid)
inp.append(out)

x = [[0],[0]]

it = iter(inp)
for l in it:
    x = l.forward(x)
print x
"""
"""
"""

"""
from iterators import LayerIterator

class Foo(object):
    def __init__(self, data, next_):
        self.next_ = next_
        self.data = data

    def __iter__(self):
        return LayerIterator(self)

foo = Foo("one", Foo("two", None))
it = iter(foo)
print next(it).data
print next(it).data
print next(it).data
"""
