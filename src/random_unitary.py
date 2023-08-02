import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis

import numpy as np

from scipy.stats import unitary_group

SPARSE_TYPE = setup.SPARSE_TYPE


def random_hamiltonian_in_subspace(nqbits: int, energy_subspace: int):
    """
    Args:
        nqbits: number of qubits
        energy_subspace: the energy subspace to generate the hamiltonian in
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy within the given subspace
    """
    # blocks = [np.array([[0]])] + [np.zeros((comb(nqbits, e), comb(nqbits, e))) if e is not energy_subspace else np.random.random((comb(nqbits, e), comb(nqbits, e))) / 2.0 * np.exp(
    #     1j * 2 * np.pi * np.random.random((comb(nqbits, e), comb(nqbits, e)))) for e in range(1, nqbits)] + [
    #              np.array([[0]])]

    blocks = [np.array([[0]])] + [
        np.zeros((comb(nqbits, e), comb(nqbits, e))) if e is not energy_subspace else np.random.random((comb(nqbits, e), comb(nqbits, e))) * 1j + np.random.random((comb(nqbits, e), comb(nqbits, e)))
        for e in range(1, nqbits)] + [
                 np.array([[0]])]

    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)
    m = (m + m.conj().T)
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(nqbits))


def random_hamiltonian(nqbits: int):
    """
    Args:
        nqbits: number of qubits
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy
    """
    blocks = [np.array([[0]])] + [np.random.random((comb(nqbits, e), comb(nqbits, e))) for e in range(1, nqbits)] + [np.array([[0]])]
    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)

    m = m + m.conj().T
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(nqbits))


def random_unitary_in_subspace(nqbits: int, energy_subspace: int, dt=1.0):
    blocks = [np.array([[1]])] + [np.eye((comb(nqbits, e))) if e is not energy_subspace else unitary_group.rvs(comb(nqbits, e)) for e in range(1, nqbits)] + [np.array([[1]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(nqbits))


def random_energy_preserving_unitary(nqbits: int, dt=1.0) -> DM.DensityMatrix:
    """
    Args:
        nqbits: the number of qbits in the system.

    Returns: A density matrix object of a random energy preserving unitary on n qbits.

    This function uses the ability to generate completely random unitaries from scipy.stats.unitary_group to generate complete random unitaries in energy preserving subspaces of the full unitary.

    """
    blocks = [np.array([[1. + 0j]])] + [unitary_group.rvs(comb(nqbits, i)) for i in range(1, nqbits)] + [np.array([[1. + 0j]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(SPARSE_TYPE(m), energy_basis(nqbits))
