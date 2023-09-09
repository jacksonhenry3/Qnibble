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
        assert all([b.shape == self.blocks[0].shape for b in self.blocks])
        return BlockSparseMatrix(self.blocks @ other.blocks)

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

    def kronecker_product(self, other):
        result_blocks = dict()
        num_blocks = len(self.blocks)

        for i in range(num_blocks):
            for j in range(num_blocks):
                # Calculate the Kronecker product between two blocks
                result_block = np.kron(self.blocks[i], other.blocks[j])
                result_blocks[(i, j)] = result_block

        # sort the blocks by the sum of their indices
        result_blocks = [result_blocks[key] for key in sorted(result_blocks.keys(), key=lambda x: x[0] + x[1])]

        # The basis is not the usual kroneger product basis !TODO
        """
        OK, so for the kronecker product the energy subspace of each block adds together. they should bre repositioned based on this.
        You can make an arbitrary choice for beyond the eergy subspace, but it should be consistent. (and needs to be iomplimented for KET aswell)
        """
        return BlockSparseMatrix(result_blocks)


eg_1 = BlockSparseMatrix([np.array([[.1]]), np.array([[.8, .8], [.8, .1]]), np.array([[.8]])])
eg_1 = eg_1.kronecker_product(eg_1)
import matplotlib.pyplot as plt

plt.imshow(eg_1.toarray())
plt.show()
# print(eg_1.toarray())
