"""

considering implimenting a class which is a numpy list of sparse matrices.


matreix mul, addition etc can be covered easily like this
print(block_diag(*list(map(lambda x, y: (x@y).toarray(), r1, r2))))



"""

import numpy as np
import numpy.typing as npt
import itertools
from math import comb

import scipy as sp
from scipy import linalg

import src.setup as setup

xp = setup.xp


class BlockSparseMatrix:

    def __init__(self, data: list[np.ndarray]):
        self.blocks = [xp.array(d) for d in data]
        dim = sum([b.shape[0] for b in self.blocks])
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
        return BlockSparseMatrix([block * other for block in self.blocks])

    def __add__(self, other):
        return BlockSparseMatrix(self.blocks + other.blocks)

    def __sub__(self, other):
        return BlockSparseMatrix([b1 - b2 for b1, b2 in zip(self.blocks, other.blocks)])

    def __neg__(self):
        return BlockSparseMatrix([-block for block in self.blocks])

    def __repr__(self):
        return self.blocks.__repr__()

    def log(self):
        return BlockSparseMatrix([np.array(sp.linalg.logm(b)) for b in self.blocks])

    def exp(self):
        return BlockSparseMatrix([np.array(sp.linalg.expm(b)) for b in self.blocks])

    def diagonal(self):
        return np.concatenate([b.diagonal() for b in self.blocks])

    def toarray(self):
        return sp.linalg.block_diag(*self.blocks)

    @property
    def H(self):
        return BlockSparseMatrix([b.conj().T for b in self.blocks])


# Utilities to generate density matrices
def Identity(n) -> BlockSparseMatrix:
    blocks = [np.array([[1. + 0j]])] + [xp.identity(comb(n, i)).astype(np.complex64) for i in range(1, n)] + [np.array([[1. + 0j]])]

    return BlockSparseMatrix(blocks)
