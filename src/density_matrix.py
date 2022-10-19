import src.setup as setup

from src.ket import energy_basis, canonical_basis, Basis

import matplotlib.pyplot as plt
from matplotlib import colors

import copy
import warnings
from functools import reduce
import numpy as np

xp = setup.xp
sp = setup.sp
SPARSE_TYPE = setup.SPARSE_TYPE

from scipy.linalg import logm


class DensityMatrix:
    __slots__ = "_data", "_basis"

    def __init__(self, matrix: SPARSE_TYPE, basis: Basis):
        """This doesn't validate inputs, eg. the basis is allowed to be wrong the dimension """
        self._data: SPARSE_TYPE = SPARSE_TYPE(matrix)
        self._basis = basis

    def __repr__(self):
        return f'DM {id(self)}'

    def __eq__(self, other):
        return self.data.shape == other.data.shape and (self.data != other.data).nnz == 0 and self.basis == other.basis

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
        result = Identity(self.basis)
        for _ in range(power):
            result *= self
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
        qbit_val = 2 ** (n-remaining_qbit-1)
        tot = 0
        diags = self.data.diagonal()
        for i, b in enumerate(self.basis):
            #This uses bitwise and to identify when the remaining_qbit qbit is 1 (b.num & qbit_val) will give false if the remaining qbit value is zero for a given state
            tot += diags[i] * bool((b.num & qbit_val))

        return qbit(float(xp.real(tot)))

    def ptrace(self, qbits: list):
        """

        Args:
            qbits: a list of indices of the qubits to trace out

        Returns: a new density matrix with the requested qubits traced out

        I don't see a way to do this sparsely? either slow calculations or reshape.

        """
        assert len(qbits) < self.basis.num_qubits, "cant completely contract"
        for q in qbits:
            assert q < self.basis.num_qubits, "qbit index out of range"

        if len(qbits) == self.basis.num_qubits - 1:
            remaining_qbit = list(set(range(self.basis.num_qubits)) - set(qbits))[0]
            return self.ptrace_to_a_single_qbit(remaining_qbit)

        n = self.number_of_qbits
        new_n = n - len(qbits)
        data = self.qbit_basis()
        axes = xp.concatenate((xp.array(qbits), xp.array(qbits) + n))
        new_data = xp.sum(data, axis=tuple(axes))
        new_data = new_data.reshape((2 ** new_n, 2 ** new_n))
        return DensityMatrix(SPARSE_TYPE(new_data), canonical_basis(n - len(qbits)))

    def _ptrace(self, qbit):

        """
        new ptrace algorithm idea

        1 reorder so qbit to be ptraced has index 0
        2: data[:n/2,:n/2]+data[n/2:,n/2:]
        3. profit"""
        n = self.number_of_qbits
        order = xp.array(range(n))
        order[0] = qbit
        order[qbit] = 0
        self.relabel_basis(order)
        self.change_to_canonical_basis()
        half_size = 2 ** (n - 1)
        new_data = self.data[:half_size, :half_size] + self.data[half_size:, half_size:]
        return DensityMatrix(new_data, canonical_basis(n - 1))

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
    def number_of_qbits(self):
        return int(xp.round(xp.log2(self.size)))

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
        idx = xp.argsort(nums)
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
        img = ax.imshow(.0001 + xp.abs(dat), interpolation='none', cmap="gist_heat", norm=colors.LogNorm())
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
            ax.grid(which='minor', color='k', linestyle='-', linewidth=.5)

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
