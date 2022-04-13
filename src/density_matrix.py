import numpy as np
from scipy import sparse

from src.ket import energy_basis, canonical_basis, Basis
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.linalg as sp
import copy
SPARSE_TYPE = sparse.csc_matrix


class DensityMatrix:
    __slots__ = "_data", "_basis"

    def __init__(self, matrix: SPARSE_TYPE, basis: Basis):
        """This doesnt validate inputs, eg. the basis is allowed to be wrong the dimension """
        self._data: SPARSE_TYPE = matrix
        self._basis = basis

    def __repr__(self):
        return f'DM {id(self)}'

    def __eq__(self, other):
        return (self.data != other.data).nnz == 0 and self.basis == other.basis

    def __add__(self, other):
        assert isinstance(other, DensityMatrix), f"Addition is only defined between two DensityMatrix objects, not {other}, of type {type(other)} and DensityMatrix"
        return DensityMatrix(self._data + other._data, self._basis)

    def __mul__(self, other):
        """Multiplication with a scalar"""
        if type(other) in [float, int, complex]:
            return DensityMatrix(self._data * other, self._basis)
        elif isinstance(other, DensityMatrix):
            assert self.basis == other.basis
            return DensityMatrix(self.data @ other.data, copy.copy(self.basis))
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __rmul__(self, other):
        if type(other) in [float, int, complex]:
            return self.__mul__(other)
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    # def __pow__(self, power: int):
    #     result = Identity(self.basis)
    #     for _ in range(power):
    #         result *= self
    #     return result

    def __neg__(self):
        return DensityMatrix(-self.data, self.basis)

    def tensor(self, *others, resultant_basis=None):
        res_data = self._data
        res_basis = self._basis
        for other in others:
            if isinstance(other, DensityMatrix):
                res_data = sparse.kron(res_data, other._data)
                if resultant_basis is None:
                    res_basis = Basis((i + j for i in res_basis for j in other._basis))
            else:
                raise TypeError(f"tensor product between {self} and {other} (type {type(other)} is not defined")
        res_basis = resultant_basis or res_basis
        return DensityMatrix(res_data, res_basis)

    def ptrace_to_a_single_qbit(self, remaining_qbit):
        tot = 0
        diags = self.data.diagonal()
        for i, b in enumerate(self.basis):
            tot += diags[i] * (b.data[remaining_qbit] == '1')

        return qbit(tot)

    # def ptrace(self, qbits: list):
    #     """
    #
    #     Args:
    #         qbits: a list of indices of the qubits to trace out
    #
    #     Returns: a new density matrix with the requested qubits traced out
    #
    #     I don't see a way to do this sparsely? either slow calculations or reshape.
    #
    #     """
    #     assert len(qbits) < self.basis.num_qubits, "cant completely contract"
    #
    #     if len(qbits) == self.basis.num_qubits - 1:
    #         remaining_qbit = list(set(range(self.basis.num_qubits)) - set(qbits))[0]
    #         return self.ptrace_to_a_single_qbit(remaining_qbit)
    #
    #     for q in qbits:
    #         assert q < self.basis.num_qubits, "qbit index out of range"
    #
    #     n = self.number_of_qbits
    #     data = self.qbit_basis()
    #     axes = np.concatenate((np.array(qbits), np.array(qbits) + n))
    #     new_data = np.sum(data, axis=tuple(axes))
    #     new_data
    #     return DensityMatrix(new_data, canonical_basis(n - len(qbits)))

    def qbit_basis(self):
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
    def number_of_qbits(self):
        return int(np.log2(self.size))

    @property
    def H(self):
        """Return the conjugate transpose of self"""
        return DensityMatrix(np.transpose(self._data).conjugate(), self._basis)

    # ==== in place modification ====

    def change_to_energy_basis(self):
        energy = [b.energy for b in self.basis]
        nums = [b.num for b in self.basis]
        idx = np.lexsort((nums, energy))
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
        img = ax.imshow(.0001 + np.abs(dat), interpolation='none', cmap="hot", norm=colors.LogNorm())
        label_list = [str(b) for b in self._basis]
        ax.set_xticks(list(range(self.size)))
        ax.set_yticks(list(range(self.size)))
        ax.set_xticks(np.arange(.5, self.size + .5), minor=True)
        ax.set_yticks(np.arange(.5, self.size + .5), minor=True)
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
            ax.grid(which='minor', color='k', linestyle='-', linewidth=0)

            plt.tick_params(
                which='major',  # Just major  ticks are affected
                left=False,  # ticks along the left edge are off
                top=False,
                labelleft=False,
                labeltop=False)  # ticks along the top edge are off

        plt.show()


# Utilities to generate density matrices
def Identity(basis: Basis) -> DensityMatrix:
    """ Creates the identity density matrix for n qubits in the energy basis"""

    return DensityMatrix(SPARSE_TYPE(np.identity(len(basis))), basis)


def qbit(pop: float) -> DensityMatrix:
    assert 0 <= pop <= .5, f"population must be between 0 and .5 but you chose {pop}"
    return DensityMatrix(SPARSE_TYPE([[1 - pop, 0], [0, pop]]), energy_basis(1))


def n_thermal_qbits(pops: list) -> DensityMatrix:
    """
    Args:
        pops: a list of population numbers between 0 and .5

    Returns:
        A density matrix for n thermal qbits with the specified populations
    """
    num_states = 2 ** len(pops)
    data = []
    for i in range(num_states):
        state = list(format(i, f'0{len(pops)}b'))
        value_list = [pops[j] if b == '1' else 1 - pops[j] for j, b in enumerate(state)]
        value = np.product(value_list)
        data.append(value)

    return DensityMatrix(sparse.diags(data, format='csc'), canonical_basis(len(pops)))


# functions that operate on density matrices
def dm_exp(dm: DensityMatrix) -> DensityMatrix:
    return DensityMatrix(sp.expm(dm.data), dm.basis)


def dm_log(dm: DensityMatrix) -> DensityMatrix:
    return DensityMatrix(SPARSE_TYPE(sp.logm(dm.data.todense())), dm.basis)


def dm_trace(dm: DensityMatrix) -> float:
    # return dm.data.trace()
    # TODO FIND A GOOD WAY TO DO THE TRACE
    return np.sum(dm.data.data)


def permute_sparse_matrix(M, new_order: list):
    """
    Reorders the rows and/or columns in a scipy sparse matrix
        using the specified array(s) of indexes
        e.g., [1,0,2,3,...] would swap the first and second row/col.
    """

    # I = np.identity(M.shape[0])
    # I = I[new_order, :]
    # I = SPARSE_TYPE(I)

    I = sparse.eye(M.shape[0]).tocoo()
    I.row = I.row[new_order]
    return I.T @ M @ I


def conserves_energy(dm: DensityMatrix) -> bool:
    dat = dm.data.toarray()
    for i in range(dm.data.shape[0]):
        for j in range(dm.data.shape[1]):
            if dat[i, j] != 0 and dm.basis[i].energy != dm.basis[j].energy:
                return False
    return True
