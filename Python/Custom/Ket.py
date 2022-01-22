import numpy as np


class Ket:
    def __init__(self, data: iter):
        self.data = data
        self._num = int(''.join([str(e) for e in data]), 2)

    def __iter__(self):
        ''' Returns the Iterator object '''
        return iter(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __repr__(self) -> str:
        return f"|{self.energy}:{''.join([['↓', '↑'][int(e)] for e in self])}⟩"

    def __lt__(self, other):
        assert isinstance(other, Ket)
        return self.energy < other.energy

    def __add__(self, other):
        return Ket(list(self) + list(other))  # THIS IS INELEGANT

    @property
    def energy(self) -> int:
        return sum([int(d) for d in self])

    @property
    def num(self) -> int:
        return self._num


class Basis(tuple[Ket]):
    pass


def canonical_basis(n):
    return Basis([Ket(list(f"{i:b}".zfill(n))) for i in range(2 ** n)])


def energy_basis(n):
    basis = canonical_basis(n)
    energy = [b.energy for b in basis]
    nums = [b.num for b in basis]
    idx = np.lexsort((nums, energy))
    return Basis(tuple(np.array(basis)[idx]))
