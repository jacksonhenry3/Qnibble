import itertools

import src.setup as setup

from src.ket import energy_basis, canonical_basis, Basis, Ket

import matplotlib.pyplot as plt
from matplotlib import colors

import copy
import warnings
from functools import reduce, lru_cache
from itertools import product
import numpy as np

xp = setup.xp
sp = setup.sp

SPARSE_TYPE = setup.SPARSE_TYPE

from scipy.sparse import coo_matrix

from scipy.linalg import logm


@lru_cache
def _ptrace_mask(n:int, basis_ints:tuple[int] , qbits: tuple[int]) -> SPARSE_TYPE:
    """Caching will work better if qbits is a sorted tuple"""
    """Returns a mask for the partial trace of a density matrix"""

    # create a numpy array of x and y indices
    x, y = np.meshgrid(basis_ints, basis_ints)


    # if qbit is an integer
    if type(qbits) == int:
        qbits = [qbits]

    masks = []
    for qbit in qbits:
        mask0 = xp.bitwise_and(~x, 2 ** (n - qbit - 1)) * xp.bitwise_and(~y, 2 ** (n - qbit - 1)) != 0
        mask1 = xp.bitwise_and(x, 2 ** (n - qbit - 1)) * xp.bitwise_and(y, 2 ** (n - qbit - 1)) != 0
        masks.append([mask0, mask1])

    all_combinations = list(product(*masks))

    mask_coordinates = []
    for combination in all_combinations:
        # bitwise and each matrix in the combination sequentially
        result = combination[0]
        for matrix in combination[1:]:
            result = xp.bitwise_and(result, matrix)
        mask_coordinates.append(np.argwhere(result))

    # rather than returning a list of boolean masks, we should try to return indices for each mask.
    return mask_coordinates


class DensityMatrix:
    __slots__ = "number_of_qbits", "_data", "_basis", "__dict__"

    def __init__(self, matrix: SPARSE_TYPE, basis: Basis):
        """This doesn't validate inputs, eg. the basis is allowed to be wrong the dimension """
        self._data: SPARSE_TYPE = SPARSE_TYPE(matrix)
        self._basis = basis
        self.number_of_qbits: int = basis.num_qubits

    def __repr__(self):
        return f'DM {id(self)}'

    def __eq__(self, other):
        rtol = 10e-5
        atol = 10e-4
        return self.data.shape == other.data.shape and np.abs(np.abs(self.data - other.data) - rtol * np.abs(other.data)).max() <= atol and self.basis == other.basis

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

    def __rmul__(self, other):
        if type(other) in [float, int, complex]:
            return self.__mul__(other)
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __pow__(self, power: int):
        # CONVERT TO DENSE MATRIX for fractional powers
        # if power is an integer use repeated multiplication
        if isinstance(power, int):
            result = self
            for _ in range(power - 1):
                result = result * self
            return result
        else:
            new_data = sp.linalg.fractional_matrix_power(self.data.toarray(), power)
            result = DensityMatrix(SPARSE_TYPE(new_data), self.basis)
            return result

    def __neg__(self):
        return DensityMatrix(-self.data, self.basis)

    def tensor(self, *others, resultant_basis=None):
        if others == tuple():
            return self
        res_data = self._data
        res_basis = self._basis
        for other in others:
            if isinstance(other, DensityMatrix):
                res_data = sp.sparse.kron(res_data, other._data)
                if resultant_basis is None:
                    res_basis = res_basis.tensor(other._basis)
            else:
                raise TypeError(f"tensor product between {self} and {other} (type {type(other)} is not defined")
        res_basis = resultant_basis or res_basis
        return DensityMatrix(res_data, res_basis)

    def ptrace_to_a_single_qbit(self, remaining_qbit):
        n = self.number_of_qbits
        qbit_val = 2 ** (n - remaining_qbit - 1)
        diags = xp.real(self.data.diagonal())
        nums = xp.array([b.num for b in self.basis])
        pop = xp.sum(diags[nums & qbit_val != 0])
        # assert that pop is between 0 and 1
        # assert 0 <= pop <= 1, f"Population of qbit {remaining_qbit} is {pop}"

        return pop

    def ptrace(self, qbits, resultant_dm = True):

        """
        trace out the given qbits
        """

        if qbits == []:
            return self


        # Add a check that all indices are valid no repeats etc.
        assert all([0 <= qbit < self.number_of_qbits for qbit in qbits]), f"qbits {qbits} are not valid for a {self.number_of_qbits} qbit system"
        # This could fail if set stops preserving order,
        assert list(set(qbits)) == list(qbits), f"qbits {qbits} are not valid for a {self.number_of_qbits} qbit system"

        # self.change_to_canonical_basis()
        basis_ints = tuple(map(lambda a: a.num, self.basis))
        masks = _ptrace_mask(self.number_of_qbits, basis_ints, qbits)
        size = 2 ** (self.number_of_qbits - len(qbits))

        # the below line is slow becouse it winds up calling ndarray.nonzero A LOT
        # new_data = np.sum([self.data[mask].reshape(size, size) for mask in masks], axis=0)

        # new_data = np.sum([self.data[np.nonzero(mask)].reshape(size, size) for mask in masks], axis=0)
        new_data_list = []
        data = self.data.toarray()
        for mask in masks:
            selected_data = data[tuple(mask.T)].reshape(size, size)
            new_data_list.append(selected_data)

        new_data = np.sum(new_data_list, axis=0)

        if not resultant_dm:
            return new_data

        basis = canonical_basis(self.number_of_qbits - len(qbits))
        result = DensityMatrix(new_data, basis)
        return result

    def qbit_basis(self) -> xp.ndarray:
        n = self.number_of_qbits
        self.change_to_canonical_basis()
        data = self.data.toarray().reshape(*[2 for _ in range(2 * n)])
        return data

    # ==== static properties ====
    @property
    def data(self) -> SPARSE_TYPE:
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

    # @property
    # def data_dense(self):
    #     return self.data.toarray()

    # ==== in place modification ====

    def change_to_energy_basis(self):
        energy = np.array([b.energy for b in self.basis])
        nums = np.array([b.num for b in self.basis])
        idx = np.lexsort(np.array([nums, energy]))
        self._data = permute_sparse_matrix(self._data, list(idx))
        self._basis = self.basis.reorder(idx)

    def change_to_canonical_basis(self):
        nums = [b.num for b in self.basis]
        # energy = [b.energy for b in self.basis]
        idx = np.argsort(nums)
        self._data = permute_sparse_matrix(self._data, idx)
        self._basis = self.basis.reorder(idx)

    def relabel_basis(self, new_order):
        """
        changes basis by changing which is the "first" qbit.

        """
        new_basis = []
        for e in self.basis:
            new_basis.append(e.reorder(new_order))
        self._basis = Basis(new_basis)

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
    return DMS[0].tensor(*DMS[1:])


# Utilities to generate density matrices
def Identity(basis: Basis) -> DensityMatrix:
    """ Creates the identity density matrix for n qubits in basis"""

    return DensityMatrix(SPARSE_TYPE(xp.identity(len(basis))), basis)


def qbit(pop: float) -> DensityMatrix:
    assert 0 <= pop <= .5, f"population must be between 0 and .5 but you chose {pop}"
    return DensityMatrix(SPARSE_TYPE(xp.array([[1 - pop, 0], [0, pop]]), dtype=xp.complex64), energy_basis(1))


def n_thermal_qbits(pops: list) -> DensityMatrix:
    """
    Args:
        pops: a list of population numbers between 0 and .5

    Returns:
        A density matrix for n thermal qbits with the specified populations
    """
    # assert pops to be between 0 and .5
    assert all([0 <= pop <= .5 for pop in pops]), f"population must be between 0 and .5 but you chose {pops}"
    num_states = 2 ** len(pops)
    data = []
    for i in range(num_states):
        state = list(format(i, f'0{len(pops)}b'))
        value_list = [pops[j] if b == '1' else 1 - pops[j] for j, b in enumerate(state)]
        value = reduce((lambda x, y: x * y), value_list)
        data.append(value)

    return DensityMatrix(sp.sparse.diags(data, format='csc'), canonical_basis(len(pops)))


# functions that operate on density matrices
def dm_exp(dm: DensityMatrix) -> DensityMatrix:
    return DensityMatrix(sp.sparse.linalg.expm(dm.data), dm.basis)


def dm_log(dm: DensityMatrix) -> DensityMatrix:
    warnings.warn("Requires conversion to and from dense", Warning)
    if xp != np:
        warnings.warn("Requires sending data to and from the gpu", Warning)
        return DensityMatrix(SPARSE_TYPE(xp.array(logm(xp.asnumpy(dm.data.todense())))), dm.basis)
    return DensityMatrix(SPARSE_TYPE(logm(dm.data.todense())), dm.basis)


def dm_trace(dm: DensityMatrix) -> float:
    return dm.data.diagonal(k=0).sum()


def permute_sparse_matrix(M, new_order: list):
    """
    Reorders the rows and/or columns in a scipy sparse matrix
        using the specified array(s) of indexes
        e.g., [1,0,2,3,...] would swap the first and second row/col.
    """

    # I = np.identity(M.shape[0])
    # I = I[new_order, :]
    # I = SPARSE_TYPE(I)

    I = sp.sparse.eye(M.shape[0], dtype=xp.float64).tocoo()
    I.row = I.row[new_order]
    I = SPARSE_TYPE(I)
    return SPARSE_TYPE(I.T @ M @ I, dtype=xp.complex64)


def permute_sparse_matrix_new(m, new_order):
    new_order = xp.array(new_order)
    coo = m.tocoo()
    row, col, data = coo.row, coo.col, coo.data

    col = new_order[col]
    row = new_order[row]

    coo = sp.sparse.coo_matrix((data, (row, col)), shape=m.shape)
    csr = coo.tocsr()
    return csr


def conserves_energy(dm: DensityMatrix) -> bool:
    dat = dm.data.toarray()
    for i in range(dm.data.shape[0]):
        for j in range(dm.data.shape[1]):
            if dat[i, j] != 0 and dm.basis[i].energy != dm.basis[j].energy:
                return False
    return True
