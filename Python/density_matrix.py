import numpy as np
from Python.ket import energy_basis, canonical_basis, Basis, Ket
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.linalg as sp


class DensityMatrix:
    def __init__(self, matrix: np.ndarray, basis: Basis):
        """This doesnt validate inputs, eg. the basis is allowed to be wrong the dimension """
        self._data: np.ndarray = matrix
        self._basis = basis

    def __repr__(self):
        return 'DM' + str(self._data)

    def __eq__(self, other):
        return np.array_equal(self.data, other.data) and self.basis == other.basis

    def __add__(self, other):
        assert isinstance(other, DensityMatrix), f"Addition is only defined between two DensityMatrix objects, not {other}, of type {type(other)} and DensityMatrix"
        return DensityMatrix(self._data + other._data, self._basis)

    def __mul__(self, other):
        """Multiplication with a scalar"""
        if type(other) in [float, int, complex]:
            return DensityMatrix(self._data * other, self._basis)
        elif isinstance(other, DensityMatrix):
            assert self.basis == other.basis
            return DensityMatrix(np.matmul(self.data, other.data), self.basis)
            # raise TypeError(f"both objects must have the same basis")
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __rmul__(self, other):
        if type(other) in [float, int, complex]:
            return self.__mul__(other)
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __pow__(self, pow: int):
        result = Identity(self.basis)
        for _ in range(pow):
            result *= self
        return result

    def __neg__(self):
        return DensityMatrix(-self.data, self.basis)

    def tensor(self, other):
        if isinstance(other, DensityMatrix):
            new_data = sp.kron(self._data, other._data)
            new_basis = Basis((i + j for i in self.basis for j in other._basis))
            res = DensityMatrix(new_data, new_basis)
            return res
        raise TypeError(f"tensor product between {self} and {other} (type {type(other)} is not defined")

    def _ptrace(self, n):
        """
        Args:
            n: the index of the qbit to be traced out

        Returns:
            A new density matrix which the nth qbit traced out (in the energy basis)

        This is incredibly slow.
        """

        num_qbits = self.basis.num_qubits

        new_num_qubits = num_qbits - 1
        new_num_states = 2 ** new_num_qubits
        new_basis = energy_basis(new_num_qubits)
        new_matrix = np.zeros((new_num_states, new_num_states), dtype=np.complex)
        for x, b1 in enumerate(new_basis):
            for y, b2 in enumerate(new_basis):
                first_X = Ket(list(b1.data)[:n] + ['0'] + list(b1.data)[n:])
                first_Y = Ket(list(b2.data)[:n] + ['0'] + list(b2.data)[n:])

                second_X = Ket(list(b1.data)[:n] + ['1'] + list(b1.data)[n:])
                second_Y = Ket(list(b2.data)[:n] + ['1'] + list(b2.data)[n:])

                first_X_i = self.basis.index(first_X)
                first_Y_i = self.basis.index(first_Y)
                second_X_i = self.basis.index(second_X)
                second_Y_i = self.basis.index(second_Y)

                new_matrix[x, y] = self.data[first_X_i, first_Y_i] + self.data[second_X_i, second_Y_i]

        return DensityMatrix(new_matrix, new_basis)

    def ptrace_legacy(self, qbits: list):
        """

        Args:
            qbits: a list of indices of the qubits to trace out

        Returns: a new density matrix with the requested qubits traced out

        """
        assert len(qbits) < self.basis.num_qubits, "cant completly contract"

        for q in qbits:
            assert q < self.basis.num_qubits, "qbit index out of range"

        result = self
        for qbit_index in sorted(qbits)[::-1]:
            result = result._ptrace(qbit_index)
        return result

    def ptrace(self, qbits: list):
        """

        Args:
            qbits: a list of indices of the qubits to trace out

        Returns: a new density matrix with the requested qubits traced out

        """
        assert len(qbits) < self.basis.num_qubits, "cant completly contract"

        for q in qbits:
            assert q < self.basis.num_qubits, "qbit index out of range"

        n = self.number_of_qbits
        data = self.qbit_basis()
        axes = np.concatenate((np.array(qbits), np.array(qbits) + n))
        new_data = np.sum(data, axis=tuple(axes))
        return DensityMatrix(new_data, canonical_basis(n - len(qbits)))

    def qbit_basis(self):
        n = self.number_of_qbits
        self.change_to_canonical_basis()
        data = self.data.reshape(*[2 for _ in range(2 * n)])
        return data

    # ==== static properties ====
    @property
    def data(self) -> np.ndarray:
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
        self._data[:, ] = self._data[:, idx]
        self._data = self._data[idx, :]
        self._basis = Basis(tuple(np.array(self._basis)[idx]))

    def change_to_canonical_basis(self):
        nums = [b.num for b in self.basis]
        # energy = [b.energy for b in self.basis]
        idx = np.argsort(nums)
        self._data[:, ] = self._data[:, idx]
        self._data = self._data[idx, :]
        self._basis = Basis(tuple(np.array(self._basis)[idx]))

    def relabel_basis(self, new_order):
        """
        changes basis by changing which is the "first" qbit.

        """
        for e in self.basis:
            e.reorder(new_order)

        self.change_to_canonical_basis()

    # ==== visualization ====

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(.0001 + np.abs(self._data), interpolation='none', cmap="hot", norm=colors.LogNorm())
        label_list = [str(b) for b in self._basis]
        ax.set_xticks(list(range(self.size)))
        ax.set_yticks(list(range(self.size)))
        ax.set_xticks(np.arange(.5, self.size + .5), minor=True)
        ax.set_yticks(np.arange(.5, self.size + .5), minor=True)
        ax.set_xticklabels(label_list)
        ax.set_yticklabels(label_list)
        ax.xaxis.tick_top()
        fig.colorbar(img)
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
class Identity(DensityMatrix):
    """ Creates the identity density matrix for n qubits in the energy basis"""

    def __init__(self, basis):
        super().__init__(np.identity(len(basis)), basis)


def qbit(pop: float) -> DensityMatrix:
    assert 0 <= pop <= .5, f"population must be between 0 and .5 but you chose {pop}"
    return DensityMatrix(np.array([[1 - pop, 0], [0, pop]]), energy_basis(1))


def nqbit(pops: list) -> DensityMatrix:
    sys = qbit(pops[0])
    for temp in pops[1:]:
        sys = sys.tensor(qbit(temp))
    sys.change_to_energy_basis()
    return sys


# functions that operate on density matrices
def dm_exp(dm: DensityMatrix) -> DensityMatrix:
    return DensityMatrix(sp.expm(dm.data), dm.basis)


def dm_log(dm: DensityMatrix) -> DensityMatrix:
    return DensityMatrix(sp.logm(dm.data), dm.basis)


def dm_trace(dm: DensityMatrix) -> float:
    return np.trace(dm.data)
