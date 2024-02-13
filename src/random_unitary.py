import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis, canonical_basis

import numpy as np

from scipy.stats import unitary_group

SPARSE_TYPE = setup.SPARSE_TYPE

from scipy.stats import rv_continuous


# code from pennylane
# https://pennylane.ai/qml/demos/tutorial_haar_measure/
class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)


# Samples of theta should be drawn from between 0 and pi
sin_sampler = sin_prob_dist(a=0, b=np.pi)


def random_hamiltonian(num_qbits: int, seed=None):
    """
    Args:
        num_qbits: number of qubits
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy
    """
    rng = np.random.default_rng(seed)

    blocks = [np.array([[0]])] + [rng.random((comb(num_qbits, e), comb(num_qbits, e))) for e in range(1, num_qbits)] + [np.array([[0]])]
    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)

    m = m + m.conj().T
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(num_qbits))


def random_hamiltonian_in_subspace(num_qbits: int, energy_subspace: int, seed=None):
    """
    Args:
        num_qbits: number of qubits
        energy_subspace: the energy subspace to generate the hamiltonian in
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy within the given subspace
    """
    rng = np.random.default_rng(seed)

    blocks = [np.array([[0]])] + [
        np.zeros((comb(num_qbits, e), comb(num_qbits, e))) if e is not energy_subspace else rng.random((comb(num_qbits, e), comb(num_qbits, e))) * 1j + rng.random(
            (comb(num_qbits, e), comb(num_qbits, e)))
        for e in range(1, num_qbits)] + [
                 np.array([[0]])]

    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)
    m = (m + m.conj().T)
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(num_qbits))


def random_hamiltonian_in_subspace_coppying_mathematica(num_qbits: int, energy_subspace: int, seed=None):
    """
    Args:
        num_qbits: number of qubits
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
    indices = [i for i in range(2 ** num_qbits) if bin(i).count("1") == energy_subspace]

    # sum over all the hamiltonians for each pair of indices
    m = sum([rng.random() * hamiltonian(indices[i1], indices[i2], num_qbits) for i1 in range(len(indices)) for i2 in range(i1)])

    result = DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), canonical_basis(num_qbits))

    result.change_to_energy_basis()

    return result


def random_unitary_in_subspace(num_qbits: int, energy_subspace: int, seed=None):
    rng = np.random.default_rng(seed)

    blocks = [np.array([[1]])] + [np.eye((comb(num_qbits, e))) if e is not energy_subspace else unitary_group.rvs(comb(num_qbits, e), random_state=rng) for e in range(1, num_qbits)] + [
        np.array([[1]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(num_qbits))


def random_energy_preserving_unitary(num_qbits: int, seed=None) -> DM.DensityMatrix:
    """
    Args:
        num_qbits: the number of qbits in the system.
        seed (optional): the seed for the random number generator.

    Returns: A density matrix object of a random energy preserving unitary on n qbits.

    This function uses the ability to generate completely random unitaries from scipy.stats.unitary_group to generate complete random unitaries in energy preserving subspaces of the full unitary.

    """

    rng = np.random.default_rng(seed)
    blocks = [np.array([[1. + 0j]])] + [unitary_group.rvs(comb(num_qbits, i), random_state=rng) for i in range(1, num_qbits)] + [np.array([[1. + 0j]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(SPARSE_TYPE(m), energy_basis(num_qbits))


def haar_random_unitary():
    """
    based off of pennylanes tutorial on haar random unitaries, this only generates a random unitary in the 2 qubit subspace
    """
    phi, omega = 2 * np.pi * np.random.uniform(size=2)  # Sample phi and omega as normal
    theta = sin_sampler.rvs(size=1)[0]  # Sample theta from our new distribution
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    data = np.array([[np.exp(-1j * (phi + omega) / 2) * c, -np.exp(1j * (phi - omega) / 2) * s],
                     [np.exp(-1j * (phi - omega) / 2) * s, np.exp(1j * (phi + omega) / 2) * c]])



    m = sp.linalg.block_diag(np.array([[1]]),data,np.array([[1]]))
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(2))