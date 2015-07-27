# system libraries
# internal libraries
# third party libraries
import numpy as np


class Cost():
    def func(self, *args):
        raise NotImplementedError
    def deriv(self, *args):
        raise NotImplementedError


class MeanSquare(Cost):
    
    # normal function defined as one
    # half of the mean square error
    def func(self, x, y):
        return 0.5 * ((x - y) ** 2)
    
    # derivative of the normal function
    def deriv(self, x, y):
        return x - y

mean_square = MeanSquare()