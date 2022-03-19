import Python.density_matrix as DM
from math import comb
from Python.ket import energy_basis
import numpy as np
from scipy.linalg import block_diag, fractional_matrix_power
from scipy.stats import unitary_group


def random_hamiltonian(nqbits: int):
    blocks_real = [np.random.random([comb(nqbits, i), comb(nqbits, i)]) for i in range(nqbits + 1)]
    blocks_complex = [1j * np.random.random([comb(nqbits, i), comb(nqbits, i)]) for i in range(nqbits + 1)]
    blocks = [blocks_real[i] + blocks_complex[i] for i in range(nqbits + 1)]
    blocks = [block + np.conjugate(block.T) for block in blocks]
    m = block_diag(*blocks)
    np.fill_diagonal(m, 0)
    return DM.DensityMatrix(m, energy_basis(nqbits))


def random_unitary(nqbits: int, dt=.01):
    blocks = [np.array([[1]])] + [unitary_group.rvs(comb(nqbits, i)) for i in range(1, nqbits)] + [np.array([[1]])]
    m = block_diag(*blocks)
    return DM.DensityMatrix(m, energy_basis(nqbits))
