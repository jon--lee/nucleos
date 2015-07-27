import nucleos
import activations
import numpy as np
import colors


def output(s=""):
    print colors.green + str(s) + colors.end


class foo(object):
    def __init__(self, data):
        self.data = data
        self.n = None

    def __iter__(self):
        return bar(self)

class bar(object):
    def __init__(self, root):
        self.root = root

    def __iter__(self):
        return self

    def next(self):
        if self.root is None:
            raise StopIteration
        else:
            temp = self.root
            self.root = self.root.n
            return temp


b = foo("hi this this b")
a = foo("hi this is a")
a.n = b
it = iter(a)
for thing in it:
    output(thing.data)