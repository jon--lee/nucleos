#
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# module that implementes the iterator interface designed
# specifically to iterate over layers which are similar to
# linked lists.
#
# for layer to implement this class add:
# 
#   def __iter__(self):
#       return LayerIterator(<iterable>)
#


# system libraries
# internal libraries
from iterator import Iterator
# third party libraries

class LayerIterator(Iterator):
    def __init__(self, root_layer):
        self.root = root_layer

    def next(self):
        if self.root is None:
            raise StopIteration
        else:
            temp = self.root
            self.root = self.root.next_
            return temp
    
    def __iter__(self):
        return self
