import numpy as np
import functools


class Ket:
    __slots__ = "data", "_num", "__dict__"

    def __init__(self, data: iter):
        self.data = np.array(data)
        self._num = int(''.join([str(e) for e in data]), 2)

    def __iter__(self):
        ''' Returns the Iterator object '''
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return list(self.data) == list(other.data)

    # this breaks putting it inside a numpy array?!
    # def __getitem__(self, item):
    #     return self.data[item]

    def __repr__(self) -> str:
        return f"|{self.num},{self.energy}:" + f"{''.join([['↓', '↑'][int(e)] for i, e in enumerate(self)])}⟩"

    def __lt__(self, other):
        assert isinstance(other, Ket)
        return self.energy < other.energy

    def __add__(self, other):
        return Ket(np.array(list(self) + list(other)))  # THIS IS INELEGANT

    @functools.cached_property
    # @property
    def energy(self) -> int:
        return sum([int(d) for d in self])

    @property
    def num(self) -> int:
        return self._num

    def reorder(self, order):
        return Ket(self.data[order])


class Basis(tuple):
    # @functools.cached_property
    @property
    def num_qubits(self):
        return int(np.log2(len(self)))

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
    return Basis([Ket(np.array(list(f"{i:b}".zfill(n)))) for i in range(2 ** n)])


@functools.lru_cache(maxsize=2 ** 12, typed=False)
def energy_basis(n):
    basis = canonical_basis(n)
    energy = [b.energy for b in basis]
    nums = [b.num for b in basis]
    idx = np.lexsort((nums, energy))
    return Basis(np.array(basis)[idx])
