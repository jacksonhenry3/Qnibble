import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis, canonical_basis

import numpy as np

from scipy.stats import unitary_group

SPARSE_TYPE = setup.SPARSE_TYPE


def random_hamiltonian_in_subspace(nqbits: int, energy_subspace: int, seed=None):
    """
    Args:
        nqbits: number of qubits
        energy_subspace: the energy subspace to generate the hamiltonian in
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy within the given subspace
    """
    rng = np.random.default_rng(seed)

    blocks = [np.array([[0]])] + [
        np.zeros((comb(nqbits, e), comb(nqbits, e))) if e is not energy_subspace else rng.random((comb(nqbits, e), comb(nqbits, e))) * 1j + rng.random((comb(nqbits, e), comb(nqbits, e)))
        for e in range(1, nqbits)] + [
                 np.array([[0]])]

    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)
    m = (m + m.conj().T)
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(nqbits))


def random_hamiltonian_in_subspace_coppying_mathematica(nqbits: int, energy_subspace: int, seed=None):
    """
    Args:
        nqbits: number of qubits
        energy_subspace: the energy subspace to generate the hamiltonian in
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy within the given subspace
    """

    rng = np.random.default_rng(seed)

    def hamiltonian(i1, i2, n):
        base = np.zeros((2 ** n, 2 ** n)) * 1j
        base[i1, i2] = 1j
        base[i2, i1] = -1j
        return base

    # get the indices of the energy subspace (i.e. all numbers with energy_subspace number of 1s in their binary representation)\
    indices = [i for i in range(2 ** nqbits) if bin(i).count("1") == energy_subspace]

    # sum over all the hamiltonians for each pair of indices
    m = sum([rng.random() * hamiltonian(indices[i1], indices[i2], nqbits) for i1 in range(len(indices)) for i2 in range(i1)])

    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), canonical_basis(nqbits))


def random_hamiltonian(nqbits: int, seed=None):
    """
    Args:
        nqbits: number of qubits
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy
    """
    rng = np.random.default_rng(seed)

    blocks = [np.array([[0]])] + [rng.random((comb(nqbits, e), comb(nqbits, e))) for e in range(1, nqbits)] + [np.array([[0]])]
    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)

    m = m + m.conj().T
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(nqbits))


def random_unitary_in_subspace(nqbits: int, energy_subspace: int, seed=None):
    rng = np.random.default_rng(seed)

    blocks = [np.array([[1]])] + [np.eye((comb(nqbits, e))) if e is not energy_subspace else unitary_group.rvs(comb(nqbits, e), seed=rng) for e in range(1, nqbits)] + [np.array([[1]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(nqbits))


def random_energy_preserving_unitary(nqbits: int, seed=None) -> DM.DensityMatrix:
    """
    Args:
        nqbits: the number of qbits in the system.
        seed (optional): the seed for the random number generator.

    Returns: A density matrix object of a random energy preserving unitary on n qbits.

    This function uses the ability to generate completely random unitaries from scipy.stats.unitary_group to generate complete random unitaries in energy preserving subspaces of the full unitary.

    """

    rng = np.random.default_rng(seed)
    blocks = [np.array([[1. + 0j]])] + [unitary_group.rvs(comb(nqbits, i), seed=rng) for i in range(1, nqbits)] + [np.array([[1. + 0j]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(SPARSE_TYPE(m), energy_basis(nqbits))