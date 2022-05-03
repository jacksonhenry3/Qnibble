import src.density_matrix as DM
from math import comb
from src.ket import energy_basis
import numpy as np
from scipy.linalg import block_diag

np.random.seed(0)
def random_hamiltonian(nqbits: int):
    blocks_real = [np.random.random([comb(nqbits, i), comb(nqbits, i)]) for i in range(nqbits + 1)]
    blocks_complex = [1j * np.random.random([comb(nqbits, i), comb(nqbits, i)]) for i in range(nqbits + 1)]
    blocks = [blocks_real[i] + blocks_complex[i] for i in range(nqbits + 1)]
    blocks = [block + np.conjugate(block.T) for block in blocks]
    m = block_diag(*blocks)
    np.fill_diagonal(m, 0)
    return DM.DensityMatrix(DM.SPARSE_TYPE(m), energy_basis(nqbits))


def random_hamiltonian_in_subspace(nqbits: int, energy_subspace: int):
    blocks_real = [np.random.random([comb(nqbits, i), comb(nqbits, i)]) if i == energy_subspace else np.identity(comb(nqbits, i)) for i in range(nqbits + 1)]
    blocks_complex = [1j * np.random.random([comb(nqbits, i), comb(nqbits, i)]) if i == energy_subspace else np.identity(comb(nqbits, i)) for i in range(nqbits + 1)]
    blocks = [blocks_real[i] + blocks_complex[i] for i in range(nqbits + 1)]
    blocks = [block + np.conjugate(block.T) for block in blocks]
    m = block_diag(*blocks)
    np.fill_diagonal(m, 0)
    return DM.DensityMatrix(DM.SPARSE_TYPE(m), energy_basis(nqbits))


def random_unitary(nqbits: int, dt=.1):
    H = random_hamiltonian(nqbits)
    return DM.dm_exp(-dt * 1j * H)


def random_unitary_in_subspace(nqbits: int, energy_subspace: int, dt=.1):
    H = random_hamiltonian_in_subspace(nqbits, energy_subspace)
    return DM.dm_exp(-dt * 1j * H)

# def random_unitary(nqbits: int, dt=.1):
#     blocks = [np.array([[1]])] + [unitary_group.rvs(comb(nqbits, i)) for i in range(1, nqbits)] + [np.array([[1]])]
#     m = block_diag(*blocks)
#     return DM.DensityMatrix(sparse.bsr_matrix(m), energy_basis(nqbits))
