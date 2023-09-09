import numpy as np
import functools


class Ket:
    def __init__(self, num, num_qbit):
        self.num = num
        self.num_qbit = num_qbit

    def __repr__(self) -> str:
        return f"|{self.num},{self.energy}:" + f"{''.join([['↓', '↑'][int(e)] for i, e in enumerate(self.data())])}⟩"

    def data(self) -> list:
        return [*bin(self.num)[2:].zfill(self.num_qbit)]

    @property
    def energy(self):
        return self.num.bit_count()

    def reorder(self, order):
        binA = bin(self.num)[2:].zfill(self.num_qbit)
        return Ket(int(''.join(binA[i] for i in order), 2), self.num_qbit)

    def __lt__(self, other):
        assert isinstance(other, Ket)
        return self.energy < other.energy

    def __add__(self, other):
        return Ket((self.num << other.num_qbit) + other.num, self.num_qbit + other.num_qbit)

    def __eq__(self, other):
        return self.num == other.num


class Basis(tuple):
    @functools.cached_property
    def num_qubits(self):
        return self[0].num_qbit

    def reorder(self, order):
        x = np.empty((len(self)), dtype=Ket)
        x[:] = self
        return Basis(tuple(x[order]))

    def __repr__(self):
        return "[" + ' '.join([str(b.num) for b in self]) + "]"

    def tensor(self, *others):
        res = self
        for other in others:
            res = Basis((i + j for i in res for j in other))
        return res


@functools.lru_cache(maxsize=2 ** 12, typed=False)
def canonical_basis(n):
    return Basis([Ket(i, n) for i in range(2 ** n)])


@functools.lru_cache(maxsize=2 ** 12, typed=False)
def energy_basis(n):
    basis = canonical_basis(n)
    energy = [b.energy for b in basis]
    nums = [b.num for b in basis]
    idx = np.lexsort((nums, energy))
    return Basis(np.array(basis)[idx])
