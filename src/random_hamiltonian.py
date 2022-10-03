import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis

import numpy as np

from scipy.stats import unitary_group

SPARSE_TYPE = setup.SPARSE_TYPE


def random_unitary_in_subspace(nqbits: int, energy_subspace: int):
    blocks = [np.array([[1]])] + [np.zeros((comb(nqbits, e), comb(nqbits, e))) if e is not energy_subspace else unitary_group.rvs(comb(nqbits, e)) for e in range(1, nqbits)] + [np.array([[1]])]
    m = sp.linalg.block_diag(*blocks)
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(nqbits))


def random_unitary(nqbits: int) -> DM.DensityMatrix:
    """
    Args:
        nqbits: the number of qbits in the system.

    Returns: A density matrix object of a random energy preserving unitary on n qbits.

    This function uses the ability to generate completely random unitaries from scipy.stats.unitary_group to generate complete random unitaries in energy preserving subspaces of the full unitary.

    """
    blocks = [np.array([[1.+0j]])] + [unitary_group.rvs(comb(nqbits, i)) for i in range(1, nqbits)] + [np.array([[1.+0j]])]
    m = sp.linalg.block_diag(*blocks)
    return DM.DensityMatrix(SPARSE_TYPE(m), energy_basis(nqbits))
