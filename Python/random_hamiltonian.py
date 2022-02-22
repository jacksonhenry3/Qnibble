import Python.density_matrix as DM
from math import comb
from Python.ket import energy_basis
import numpy as np
from scipy.linalg import block_diag


def random_hamiltonian(nqbits: int):
    blocks = [.1 * np.random.random([comb(nqbits, i), comb(nqbits, i)]) for i in range(nqbits + 1)]
    m = block_diag(*blocks)
    np.fill_diagonal(m, 0)
    return DM.DensityMatrix(m, energy_basis(nqbits))
