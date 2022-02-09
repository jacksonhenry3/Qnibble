import numpy as np
from Ket import energy_basis, canonical_basis, Basis
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.linalg as sp


class DensityMatrix:
    def __init__(self, matrix, basis):
        self._data = matrix
        self._basis = basis

    def __repr__(self):
        return 'DM' + str(self._data)

    def __eq__(self, other):
        return (self.data == other.data).all() and self.basis == other.basis

    def __add__(self, other):
        assert isinstance(other, DensityMatrix), f"multiplication is only defined between two DensityMatrix objects, not {other}, of type {type(other)} and DensityMatrix"
        return DensityMatrix(self._data + other._data, self._basis)

    def tensor(self, other):
        if isinstance(other, DensityMatrix):
            new_data = sp.kron(self._data, other._data)
            new_basis = Basis((i + j for i in self.basis for j in other._basis))
            res = DensityMatrix(new_data, new_basis)
            # res.change_to_energy_basis()
            return res
        raise TypeError(f"tensor product between {self} and {other} (type {type(other)} is not defined")

    def __mul__(self, other):
        """Multiplication with a scalar"""
        if type(other) in [float, int, complex]:
            return DensityMatrix(self._data * other, self._basis)
        elif isinstance(other, DensityMatrix):
            if self.basis == other.basis:
                return DensityMatrix(np.matmul(self.data, other.data), self.basis)
            raise TypeError(f"both objects must have the same basis")
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __rmul__(self, other):
        if type(other) in [float, int, complex]:
            return self.__mul__(other)
        raise TypeError(f"multiplication between {self} and {other} (type {type(other)} is not defined")

    def __pow__(self, pow: int):
        result = self
        for _ in range(pow):
            result *= self
        return result

    # ==== static properties ====
    @property
    def data(self):
        return self._data

    @property
    def basis(self):
        return self._basis

    @property
    def size(self):
        return self._data.shape[0]

    @property
    def H(self):
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
        energy = [b.energy for b in self.basis]
        idx = np.lexsort((energy, nums))
        self._data[:, ] = self._data[:, idx]
        self._data = self._data[idx, :]
        self._basis = Basis(tuple(np.array(self._basis)[idx]))

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


# labelbottom, labeltop, labelleft, labelright

class Identity(DensityMatrix):
    """ Creates the identity density matrix for n qubits in the energy basis"""

    def __init__(self, n_qbits):
        super().__init__(np.identity(2 ** n_qbits), energy_basis(n_qbits))


def qbit(temp: float) -> DensityMatrix:
    return DensityMatrix(np.array([[1 - temp, 0], [0, temp]]), energy_basis(1))


def nqbit(temps: list) -> DensityMatrix:
    sys = qbit(temps[0])
    for temp in temps[1:]:
        sys = sys.tensor(qbit(temp))
    sys.change_to_energy_basis()
    return sys


def exp(dm: DensityMatrix)-> DensityMatrix:
    print(dm.basis == energy_basis(6))
    new_dm = DensityMatrix(sp.expm(dm.data), dm.basis)
    print(new_dm.basis == energy_basis(6))
    return new_dm
