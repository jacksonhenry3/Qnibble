from src.ket import energy_basis
from src.density_matrix import DensityMatrix, Identity, dm_exp
import itertools


def step(dm: DensityMatrix, groups: list[list[int]], Hamiltonians: list[DensityMatrix], step_sizes: list[float]) -> DensityMatrix:
    """

    Args:
        dm: the density matrix to evolve
        groups: a list of each grouping of qbits by index. e.g. [[0,1],[2,3]]
        Hamiltonians: A list of Hamiltonians that will be used to evolve each sub group
        step_sizes: what dθ should be used for each group

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    assert len(groups) == len(Hamiltonians) == len(step_sizes)

    # make sure that the given hamiltonians have the same sizes as the given groupings
    for index, hamiltonian in enumerate(Hamiltonians): assert hamiltonian.number_of_qbits == len(groups[index])

    order = list(itertools.chain.from_iterable(groups))

    # make sure each qbit is assigned to a group and that there are no extras or duplicates.
    assert set(order) == set(range(dm.number_of_qbits))

    # Generate a list of unitaries that are applied to the entire density matrix but only act on each sub group.

    # this assumes that the hamiltonians are given in the energy basis and will fail otherwise.
    identities = [Identity(energy_basis(len(group))) for group in groups]
    Unitarys = []

    for index in range(len(groups)):
        to_tensor = identities.copy()
        to_tensor[index] = Hamiltonians[index]
        H = to_tensor[0].tensor(*to_tensor[1:])
        U = dm_exp(-1j * H * step_sizes[index])
        U.relabel_basis(order)
        U.change_to_energy_basis()
        Unitarys.append(U)

    # Apply each unitary to the density matrix
    for U in Unitarys:
        dm = U * dm * U.H

    return dm
