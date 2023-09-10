"""

considering implimenting a class which is a numpy list of sparse matrices.


matreix mul, addition etc can be covered easily like this
print(block_diag(*list(map(lambda x, y: (x@y).toarray(), r1, r2))))



"""

import numpy as np
import numpy.typing as npt
import itertools

import scipy as sp
from scipy import linalg


class BlockSparseMatrix:

    def __init__(self, data: npt.NDArray[object] or list):
        self.blocks = np.array(data, dtype=object)
        dim = sum([b.shape[0] for b in data])
        self.shape = (dim, dim)

    def __matmul__(self, other):
        # assert that each block is the same size
        assert all([self.blocks[i].shape == other.blocks[i].shape for i in range(len(self.blocks))])
        result = []
        for i in range(len(self.blocks)):
            result.append(self.blocks[i] @ other.blocks[i])
        return BlockSparseMatrix(result)

    def __mul__(self, other):
        assert type(other) is int or float
        return BlockSparseMatrix(self.blocks * other)

    def __add__(self, other):
        return BlockSparseMatrix(self.blocks + other.blocks)

    def __sub__(self, other):
        return BlockSparseMatrix(self.blocks - other.blocks)

    def __repr__(self):
        return self.blocks.__repr__()

    def diagonal(self):
        return [b.diagonal() for b in self.blocks]

    def toarray(self):
        return sp.linalg.block_diag(*self.blocks)

    @property
    def H(self):
        return BlockSparseMatrix([b.conj().T for b in self.blocks])
