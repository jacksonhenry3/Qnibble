import itertools

from scipy.linalg import logm

import src.setup as setup

from src.Block_Sparse_Matrix import BlockSparseMatrix as BSM

from src.ket import energy_basis, canonical_basis, Basis, Ket

import matplotlib.pyplot as plt
from matplotlib import colors
import copy
import warnings
from functools import reduce, lru_cache
from math import comb
import numpy as np

xp = setup.xp
sp = setup.sp

import scipy.special


# from scipy.linalg import logm


# @lru_cache
# def _ptrace_mask(n: int, qbits: list[int]) -> BSM:
#     """Returns a mask for the partial trace of a density matrix"""
#
#     # create a numpy array of x and y indices
#     x, y = xp.indices((2 ** n, 2 ** n))
#
#     # if qbit is an integer
#     if type(qbits) == int:
#         qbit = qbits
#         mask0 = BSM(xp.bitwise_and(~x, 2 ** (n - qbit - 1)) * xp.bitwise_and(~y, 2 ** (n - qbit - 1)) != 0)
#         mask1 = BSM(xp.bitwise_and(x, 2 ** (n - qbit - 1)) * xp.bitwise_and(y, 2 ** (n - qbit - 1)) != 0)
#
#         return [mask0, mask1]
#
#     masks = []
#     # loop over all qbits and
#     for qbit in qbits:
#         print(qbit)
#         masks.append(_ptrace_mask(n, qbit))
#
#     res = itertools.product(*masks)
#     res = [reduce(lambda x, y: x * y, r) for r in res]
#     return res


class DensityMatrix:
    __slots__ = "number_of_qbits", "_data", "_basis", "__dict__"

    def __init__(self, matrix: BSM, basis: Basis):
        """This doesn't validate inputs, eg. the basis is allowed to be wrong the dimension """

        self._data: BSM = matrix
        self._basis = basis
        self.number_of_qbits: int = basis.num_qubits

    def __repr__(self):
        return f'DM {id(self)}'

    def __eq__(self, other):

        if not isinstance(other, DensityMatrix):
            return False
        if self.number_of_qbits != other.number_of_qbits:
            return False
        if self.basis != other.basis:
            return False
        for b1, b2 in zip(self.data.blocks, other.data.blocks):
            if not np.allclose(b1, b2):
                return False
        return True

    def __add__(self, other):
        assert isinstance(other, DensityMatrix), f"Addition is only defined between two DensityMatrix objects, not {other}, of type {type(other)} and DensityMatrix"
        assert self.basis == other.basis
        return DensityMatrix(self._data + other._data, self._basis)

    def __mul__(self, other):
        """Multiplication with a scalar"""
        if type(other) in [float, int, complex]:
            return DensityMatrix(self._data * other, self._basis)
        elif isinstance(other, DensityMatrix):
            assert self.basis == other.basis
            # TODO figure out why this coppy is needed and remove it
            return DensityMatrix(self.data @ other.data, copy.copy(self.basis))
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __neg__(self):
        return DensityMatrix(-self.data, self.basis)

    def tensor(self, other):
        if not isinstance(other, DensityMatrix):
            raise TypeError(f"tensor product between {self} and {other} (type {type(other)} is not defined")

        result_blocks = dict()
        self_num_blocks = len(self.data.blocks)
        other_num_blocks = len(other.data.blocks)

    def __rmul__(self, other):
        if type(other) in [float, int, complex]:
            return self.__mul__(other)
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __pow__(self, power: int):
        result = Identity(self.basis)
        for _ in range(power):
            result *= self
        return result

    def __neg__(self):
        return DensityMatrix(-self.data, self.basis)

    def tensor(self, other):
        if not isinstance(other, DensityMatrix):
            raise TypeError(f"tensor product between {self} and {other} (type {type(other)} is not defined")

        result_blocks = dict()
        self_num_blocks = len(self.data.blocks)
        other_num_blocks = len(other.data.blocks)
        basis_1_index = 0

        for i in range(self_num_blocks):
            basis_2_index = 0
            for j in range(other_num_blocks):
                # Calculate the Kronecker product between two blocks
                result_block = xp.kron(self.data.blocks[i], other.data.blocks[j])

                self_sub_space_basis = Basis(self.basis[basis_1_index:basis_1_index + self.data.blocks[i].shape[0]])
                other_sub_space_basis = Basis(other.basis[basis_2_index:basis_2_index + other.data.blocks[j].shape[0]])

                result_sub_space_basis = self_sub_space_basis.tensor(other_sub_space_basis)

                result_blocks[(i, j)] = result_block, result_sub_space_basis

                basis_2_index += other.data.blocks[j].shape[0]
            basis_1_index += self.data.blocks[i].shape[0]

        # sort the blocks by the sum of their indices
        result_blocks_sorted = {key: result_blocks[key] for key in sorted(result_blocks.keys(), key=lambda x: x[0] + x[1])}

        energy_blocks = dict()

        for energy, group in itertools.groupby(result_blocks_sorted, key=lambda x: x[0] + x[1]):
            energy_block = []
            energy_basis = []
            for key in group:
                block, basis = result_blocks[key]
                energy_block.append(block)
                energy_basis += list(basis)
            energy_blocks[energy] = sp.linalg.block_diag(*energy_block), energy_basis

        # sort the basis of each block in final_blocks by the basis number

        for energy in energy_blocks:
            basis = energy_blocks[energy][1]
            idx = np.lexsort(np.array([[b.num for b in basis]]))
            reordered_matrix = energy_blocks[energy][0][idx]
            reordered_matrix = reordered_matrix[:, idx]
            energy_blocks[energy] = reordered_matrix, [basis[i] for i in idx]

        final_blocks = []
        final_basis = []
        for energy in sorted(energy_blocks.keys()):
            final_blocks.append(energy_blocks[energy][0].astype(xp.complex64))
            final_basis += energy_blocks[energy][1]

        return DensityMatrix(BSM(final_blocks), Basis(final_basis))

    # @profile
    def ptrace_to_a_single_qbit(self, remaining_qbit):
        n = self.number_of_qbits
        qbit_val = 2 ** (n - remaining_qbit - 1)
        diags = xp.real(self.data.diagonal())
        nums = xp.array([b.num for b in self.basis])
        pop = xp.sum(diags[nums & qbit_val != 0])
        # assert that pop is between 0 and 1
        # assert 0 <= pop <= 1, f"Population of qbit {remaining_qbit} is {pop}"
        return pop

    def ptrace(self, qbits):
        # Add a check that all indices are valid no repeats etc.
        result = self
        for qbit_index in sorted(qbits)[::-1]:
            result = result._ptrace(qbit_index)
        return result

    def _ptrace(self, qbit):

        # TODO create composite masks for multiple qbits

        n = self.number_of_qbits

        # create a numpy array of x and y indices
        x, y = xp.indices((2 ** n, 2 ** n))
        mask0 = xp.bitwise_and(~x, 2 ** (n - qbit - 1)) * xp.bitwise_and(~y, 2 ** (n - qbit - 1)) != 0
        mask1 = xp.bitwise_and(x, 2 ** (n - qbit - 1)) * xp.bitwise_and(y, 2 ** (n - qbit - 1)) != 0

        half_size = 2 ** (n - 1)

        new_data = self.data[mask0].reshape(half_size, half_size) + self.data[mask1].reshape(half_size, half_size)
        res = DensityMatrix(new_data, canonical_basis(n - 1))
        return res

    def qbit_basis(self) -> xp.ndarray:
        n = self.number_of_qbits
        self.change_to_canonical_basis()
        data = self.data.toarray().reshape(*[2 for _ in range(2 * n)])
        return data

    # ==== static properties ====
    @property
    def data(self) -> BSM:
        return self._data

    @property
    def basis(self):
        return self._basis

    @property
    def size(self):
        return self._data.shape[0]

    @property
    def H(self):
        """Return the conjugate transpose of self"""
        return DensityMatrix(self._data.H, self._basis)

    def relabel_basis(self, new_order):
        """
        changes basis by changing which is the "first" qbit.

        """
        new_basis = []
        for e in self.basis:
            new_basis.append(e.reorder(new_order))
        # self._basis = Basis(new_basis)

        block_sizes = [b.shape[0] for b in self.data.blocks]
        current_index = 0

        # loop over each energy block
        for i, block in enumerate(self.data.blocks):
            sub_basis = new_basis[current_index:current_index + block_sizes[i]]

            current_index += block_sizes[i]

            # find the new order of the basis BELOW IS WRONG
            new_order = [b.num for b in sub_basis]
            new_order = np.argsort(new_order)

            # reorder the block
            self.data.blocks[i] = self.data.blocks[i][new_order]
            self.data.blocks[i] = self.data.blocks[i][:, new_order]

    # ==== visualization ====

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        dat = self._data.toarray()
        img = ax.imshow(.0001 + np.abs(dat), interpolation='none', cmap="gist_heat", norm=colors.LogNorm())
        label_list = [str(b) for b in self._basis]
        ax.set_xticks(list(range(self.size)))
        ax.set_yticks(list(range(self.size)))
        ax.set_xticks(xp.arange(.5, self.size + .5), minor=True)
        ax.set_yticks(xp.arange(.5, self.size + .5), minor=True)
        ax.set_xticklabels(label_list)
        ax.set_yticklabels(label_list)
        ax.xaxis.tick_top()
        # fig.colorbar(img)
        plt.xticks(rotation=75)
        plt.tick_params(
            which='major',  # Just major  ticks are affected
            left=False,  # ticks along the left edge are off
            top=False)  # ticks along the top edge are off
        if self.size < 2 ** 6:
            ax.grid(which='minor', color='k', linestyle='-', linewidth=1.5)
        else:
            # ax.grid(which='minor', color='k', linestyle='-', linewidth=.5)

            plt.tick_params(
                which='major',  # Just major  ticks are affected
                left=False,  # ticks along the left edge are off
                top=False,
                labelleft=False,
                labeltop=False)  # ticks along the top edge are off

        plt.show()


def tensor(DMS: list[DensityMatrix]) -> DensityMatrix:
    """An alias to tensor together a list of density matrices"""
    result = DMS[0]
    for dm in DMS[1:]:
        result = result.tensor(dm)
    return result


# Utilities to generate density matrices
def Identity(basis: Basis) -> DensityMatrix:
    """ Creates the identity density matrix for n qubits in basis"""

    nqbits = basis.num_qubits

    # construct the ap[roapriate blocks
    blocks = [np.array([[1. + 0j]])] + [xp.identity(comb(nqbits, i)).astype(np.complex64) for i in range(1, nqbits)] + [np.array([[1. + 0j]])]

    return DensityMatrix(BSM(blocks), basis)


def qbit(pop: float) -> DensityMatrix:
    assert 0 <= pop <= .5, f"population must be between 0 and .5 but you chose {pop}"
    return DensityMatrix(BSM([xp.array([[1 - pop]]), xp.array([[pop]])]), energy_basis(1))


def n_thermal_qbits(pops: list) -> DensityMatrix:
    """
    Args:
        pops: a list of population numbers between 0 and .5

    Returns:
        A density matrix for n thermal qbits with the specified populations
    """
    # assert pops to be between 0 and .5
    assert all([0 <= pop <= .5 for pop in pops]), f"population must be between 0 and .5 but you chose {pops}"
    N = len(pops)
    num_states = 2 ** N
    data = []
    basis = energy_basis(N)
    for i in basis:
        state = list(format(i.num, f'0{len(pops)}b'))
        value_list = [pops[j] if b == '1' else 1 - pops[j] for j, b in enumerate(state)]
        value = reduce((lambda x, y: x * y), value_list)
        data.append(value)

    result = []
    index = 0
    for energy_subspace in range(N + 1):
        block_size = scipy.special.comb(N, energy_subspace, exact=True)  # If this is slow use binom and round
        result.append(np.diag(data[index:index + block_size]))
        index += block_size

    return DensityMatrix(BSM(result), basis)


# functions that operate on density matrices
def dm_exp(dm: DensityMatrix) -> DensityMatrix:
    return DensityMatrix(dm.data.exp(), dm.basis)


def dm_log(dm: DensityMatrix) -> DensityMatrix:
    if xp != np:
        warnings.warn("Requires sending data to and from the gpu", Warning)
        raise NotImplementedError
        # return DensityMatrix(BSM(xp.array(logm(xp.asnumpy(dm.data.todense())))), dm.basis)
    return DensityMatrix(dm.data.log(), dm.basis)


def dm_trace(dm: DensityMatrix) -> float:
    return dm.data.diagonal().sum()


def conserves_energy(dm: DensityMatrix) -> bool:
    dat = dm.data.toarray()
    for i in range(dm.data.shape[0]):
        for j in range(dm.data.shape[1]):
            if dat[i, j] != 0 and dm.basis[i].energy != dm.basis[j].energy:
                return False
    return True
