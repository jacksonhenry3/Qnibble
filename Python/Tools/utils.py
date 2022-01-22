import numpy as np
import qutip as q
from scipy.linalg import logm as sp_logm


def random_complex() -> np.complex:
    """Generates a random complex number with the real and complex components between zero and one"""
    return np.random.rand() + 1j * np.random.rand()


def logm(dm: q.Qobj) -> q.Qobj:
    """Matrix log of quantum operator.

    Input operator must be square.

    Parameters
    ----------
    dm a square density matrix


    Returns
    -------
    oper : :class:`qutip.Qobj`
        Exponentiated quantum operator.

    Raises
    ------
    TypeError
        Quantum operator is not square.

    """
    if dm.dims[0][0] != dm.dims[1][0]:
        raise TypeError('Invalid operand for matrix log')

    F = sp_logm(dm.full())

    out = q.Qobj(F, dims=dm.dims)
    return out.tidyup()


def I(n):
    """Returns the n qubit identity operator"""
    return q.identity([2 for _ in range(n)])


def inverse_order(ord: list) -> list:
    """Gives the ordering the will undo ord.

    if a Qobj has been permuted by ord, permuting by inverse_order will give backj the unpermuted Qobj
    """
    return [ord.index(i) for i in range(len(ord))]


def n_thermal_qubits(temps):
    """takes the tensor product of n thermal qubits at temps"""
    return q.tensor([q.thermal_dm(2, t, "analytic") for t in temps])
