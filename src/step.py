from src.ket import energy_basis, canonical_basis
from src.density_matrix import DensityMatrix, Identity, dm_exp
import itertools
import numpy as np

from src import measurements


def step(dm: DensityMatrix, groups: list[list[int]], Unitarys: list[DensityMatrix]) -> DensityMatrix:
    """

    Args:
        dm: the density matrix to evolve
        groups: a list of each grouping of qbits by index. e.g. [[0,1],[2,3]]
        Unitarys: A list of Unitarys that will be used to evolve each sub group

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    assert len(groups) == len(Unitarys)

    # make sure that the given hamiltonians have the same sizes as the given groupings
    for index, unitary in enumerate(Unitarys): assert unitary.number_of_qbits == len(groups[index])

    temp_order = np.array(list(itertools.chain.from_iterable(groups)))
    order = ['error' for _ in range(len(temp_order))]
    for i,index in enumerate(temp_order):
        order[index] = i

    # make sure each qbit is assigned to a group and that there are no extras or duplicates.
    assert set(order) == set(range(dm.number_of_qbits))

    # Generate a list of unitaries that are applied to the entire density matrix but only act on each sub group.

    # this assumes that the hamiltonians are given in the energy basis and will fail otherwise.
    identities = [Identity(canonical_basis(len(group))) for group in groups]
    total_unitarys = []

    for index in range(len(groups)):
        to_tensor = identities.copy()
        to_tensor[index] = Unitarys[index]
        U = to_tensor[0].tensor(*to_tensor[1:])
        U.relabel_basis(order)
        U.change_to_energy_basis()
        total_unitarys.append(U)

    # Apply each unitary to the density matrix
    for U in total_unitarys:
        dm = U * dm * U.H

    return dm
