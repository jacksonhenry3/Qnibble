import src.density_matrix as DM
import itertools
import numpy as np

def step(dm: DM.DensityMatrix, groups: list[list[int]], Unitarys: list[DM.DensityMatrix]) -> DM.DensityMatrix:
    """

    Args:
        dm: the density matrix to evolve
        groups: a list of each grouping of qbits by index. e.g. [[0,1],[2,3]]
        Unitarys: A list of Unitarys that will be used to evolve each sub group

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    assert len(groups) == len(Unitarys), "Must have a unitary for every group"

    # make sure that the given hamiltonians have the same sizes as the given groupings
    for index, unitary in enumerate(Unitarys): assert unitary.number_of_qbits == len(groups[index]), "Each unitary must be for a system of the size of each group"

    order = np.array(list(itertools.chain.from_iterable(groups)))

    # make sure each qbit is assigned to a group and that there are no extras or duplicates.
    assert set(order) == set(range(dm.number_of_qbits))

    # find the product
    U = DM.tensor([Unitary for Unitary in Unitarys])

    U.relabel_basis(order)
    U.change_to_energy_basis()
    dm.change_to_energy_basis()
    dm = U * dm * U.H

    return dm
