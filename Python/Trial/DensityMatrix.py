from scipy.sparse import coo_matrix, block_diag
import numpy as np
from math import comb
from dataclasses import dataclass
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy import linalg


class DensityMatrix:
    def __init__(self, matrix, basis):
        self.data = matrix
        self.basis = basis
        # self.size =

    @property
    def size(self):
        return self.data.shape[0]

    def __add__(self, other):
        return DensityMatrix(self.data + other._data, self.basis)

    def __mul__(other,self):
        return DensityMatrix(self._data * other, self._basis)

    def __rmul__(self, other):
        return DensityMatrix(self.data * other, self.basis)

    @property
    def T(self):
        return DensityMatrix(np.transpose(self.data), self.basis)

    def change_to_energy_basis(self):
        pass
        idx = np.argsort([b.energy for b in self.basis])
        self.data[:, ] = self.data[:, idx]
        self.data = self.data[idx, :]
        self.basis = np.array(self.basis)[idx]

    def change_to_canonical_basis(self):
        pass

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(np.abs(self.data), interpolation='none', cmap="hot")
        label_list = [str(b) for b in self.basis]
        ax.set_xticks(list(range(self.size)))
        ax.set_yticks(list(range(self.size)))
        ax.set_xticks(np.arange(.5, self.size + .5), minor=True)
        ax.set_yticks(np.arange(.5, self.size + .5), minor=True)
        ax.set_xticklabels(label_list)
        ax.set_yticklabels(label_list)
        ax.xaxis.tick_top()
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1.5)
        # fig.colorbar(img)
        plt.xticks(rotation=75)
        plt.show()


@dataclass(frozen=True)
class Ket:
    data: list

    def __repr__(self) -> str:
        return f"|{self.energy},{''.join([str(e) for e in self.data])}⟩"

    def __add__(self, other):
        return Ket(self.data + other._data)

    @property
    def energy(self) -> int:
        return sum([int(d) for d in self.data])


def nqubit(n) -> coo_matrix:
    return block_diag([.1 * np.random.random([comb(n, i), comb(n, i)]) for i in range(n + 1)])


def qbt(t):
    return np.array([[t, 0], [0, 1 - t]])


# f"{n:04b}" remove the 4? add the zfill from 78
def sum_binary_digits(n): return sum([int(d) for d in f"{n:04b}"])


def energy_basis(n):
    lst = list(range(2 ** n))
    lst.sort(key=sum_binary_digits)
    return [Ket(f"{i:b}".zfill(n)) for i in lst]


def tensor(*args):
    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    elif len(args) == 1 and isinstance(args[0], Qobj):
        # tensor is called with a single Qobj as an argument, do nothing
        return args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args

    out = DensityMatrix(np.array([]), np.array([]))
    for n, q in enumerate(qlist):
        if n == 0:
            out.data = q._data
            out.basis = q._basis
        else:
            out.data = sp.kron(out.data, q._data, format='csr')
            out.basis = [i + j for i in out.basis for j in q._basis]
    out.data = out.data.toarray()
    return out


qb = DensityMatrix(qbt(.25), energy_basis(1))

n = 8
m = 4
dm = tensor([qb for i in [0] * n])
dm.change_to_energy_basis()

for t in range(1000):
    print(t)
    H1 = DensityMatrix(block_diag([.1 * np.random.random([comb(n, i), comb(n, i)]) for i in range(m + 1)]).toarray(), energy_basis(m))
    H2 = DensityMatrix(block_diag([.1 * np.random.random([comb(n, i), comb(n, i)]) for i in range(m + 1)]).toarray(), energy_basis(m))
    I = DensityMatrix(np.identity(2 ** m), energy_basis(m))
    H = tensor(H1, I) + tensor(I, H2)
    H = H + H.T
    U = linalg.expm((1j * 1. * H).data)
    dm = DensityMatrix(np.matmul(dm.data, U), energy_basis(n))
# plt.show()
dm.plot()
# t = DensityMatrix(nqubit(3), energy_basis(3))
# t.plot()
# np.trace(t.data.reshape(n,m,n,m), axis1=1, axis2=3)


# https://scicomp.stackexchange.com/questions/27496/calculating-partial-trace-of-array-in-numpy
