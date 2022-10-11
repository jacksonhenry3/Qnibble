import numpy as np
import os

import src.density_matrix as DM
from src.random_unitary import random_unitary
import copy


def run(dm: DM.DensityMatrix, measurement_set, num_iterations: int, num_chunks: int, orders, Unitaries=None, verbose=False):
    """
    Args:
        dm: the density matrix to evolve
        measurement_set: a list of functions that take a density matrix as an argument and return a number
        num_iterations: An integer representing the number of iterations the system will go through
        num_chunks : The number of subsystems that the full system will be broken in to.
        Unitaries: either a list of DMs to be used to evolve the system, if there are fewer unitaries than iterations they will be used cyclically.
                       or: a single unitary to be used at each step
                       or: None, in which case random unitaries will be generated at each step.
        orders: A list of qbit orders at each iteration step. If there are fewer orders than steps they will be used cyclically.
        verbose: a float or false. if it is a float between zero and 1 progress will be reported every verbose percent. i.e verbose =.1 will give ten progress reports

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    if type(measurement_set) != list:
        measurement_set = [measurement_set]

    measurement_values = [np.array(measurement(dm)) for measurement in measurement_set]

    generate_random_unitary = False

    if type(Unitaries) == list:
        assert len(Unitaries) == num_iterations, "There must be a unitary for each trial"
        num_unitaries = len(Unitaries)

    elif type(Unitaries) == DM.DensityMatrix:
        Unitaries = [Unitaries]
        num_unitaries = 1
    else:
        generate_random_unitary = True
        print("using random unitaries")

    chunk_size = dm.number_of_qbits // num_chunks
    leftovers = dm.number_of_qbits-chunk_size*num_chunks
    if leftovers:
        leftover_identity = DM.Identity(DM.energy_basis(leftovers))

    for i in range(num_iterations):
        progress = i / num_iterations
        if int(progress * 100) % int(verbose * 100) == 0:
            print(progress)

        order = orders[i % len(orders)]

        if generate_random_unitary:

            U = DM.tensor([random_unitary(chunk_size) for _ in range(num_chunks)])

            if leftovers:
                U = U.tensor(leftover_identity)
        else:
            U = Unitaries[i % num_unitaries]

        dm = step(dm, order, U, not generate_random_unitary)

        measurement_values = [np.vstack((measurement_values[i], measurement(dm))) for i, measurement in enumerate(measurement_set)]

    return measurement_values


def step(dm: DM.DensityMatrix, order: list[int], Unitary: DM.DensityMatrix, unitary_reused=False) -> DM.DensityMatrix:
    """
    Args:
        dm: the density matrix to evolve
        order: the qbit order to be used e.g. [0,2,1,3]
        Unitary: A Unitary that will be used to evolve the system
        unitary_reused: if the unitary will be reused make sure to undo the reordering

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    # Unitary = copy.deepcopy(Unitary)
    # make sure each qbit is assigned to a group and that there are no extras or duplicates.
    assert set(order) == set(range(dm.number_of_qbits)), f"{set(order)} vs {set(range(dm.number_of_qbits))}"
    Unitary.relabel_basis(order)
    Unitary.change_to_energy_basis()
    dm.change_to_energy_basis()
    dm = Unitary * dm * Unitary.H

    if unitary_reused:
        inverse_order = list(range(len(order)))
        for i, value in enumerate(order):
            inverse_order[value] = i
        Unitary.relabel_basis(inverse_order)

    return dm


def save_data(data: np.ndarray, num_qbits: str, measurement: str, num_chunks: str, connectivity_type: str, run_index: str, sim_index=int, extra=""):
    if extra != "":
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}_other={extra}_index={sim_index}"
    else:
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}_index={sim_index}"
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = path + f"/{measurement}_{run_index}.dat"
    np.savetxt(file_name, data, header=f"{measurement} for {num_qbits} qbits with connectivity {connectivity_type} in chunks {num_chunks}")
