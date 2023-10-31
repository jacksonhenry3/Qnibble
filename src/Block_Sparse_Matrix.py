"""

considering implimenting a class which is a numpy list of sparse matrices.


matreix mul, addition etc can be covered easily like this
print(block_diag(*list(map(lambda x, y: (x@y).toarray(), r1, r2))))



"""

import numpy as np
import scipy.sparse as s
from density_matrix import SPARSE_TYPE
import numpy.typing as npt
import itertools


class BlockDiagonalSparse:

    def __init__(self, data: npt.NDArray[SPARSE_TYPE]):
        self.data = data

    def __matmul__(self, other):
        return BlockDiagonalSparse(self.data @ other.data)

    def __mul__(self, other):
        assert type(other) is int or float
        return BlockDiagonalSparse(self.data * other)

    def __add__(self, other):
        return BlockDiagonalSparse(self.data + other.data)

    def __sub__(self, other):
        return BlockDiagonalSparse(self.data - other.data)

    def __repr__(self):
        return self.data.__repr__()

    def toarray(self):
        return [b.toarray() for b in self.data]

    """HIGLY SPECULATIVE"""
    """https://math.stackexchange.com/questions/3346742/kronecker-product-of-two-block-diagonal-matrices"""

    def kronecker_product(self, other):
        pass




BDS = BlockDiagonalSparse

eg_1 = BDS(np.array([s.eye(3, 3), s.eye(3, 3), s.eye(3, 3), s.eye(3, 3)]))
eg_2 = BDS(np.array([s.eye(3, 3), s.eye(3, 3), s.eye(3, 3), s.eye(3, 3)]))

# print(eg_1.toarray()@eg_1.toarray())
print(eg_1 @ eg_2)


# def prime_factors(n):
#   prime_list= []
#   i = 2
#   while n>1 :
#     if n%i == 0:
#       prime_list.append(i)
#       n = n/i
#       i = 2
#     else:
#         i+=1
#   return(prime_list)
#
# print(prime_factors(100))