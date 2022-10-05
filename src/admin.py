import numpy as np
import os

import src.density_matrix as DM
from src.step import step
from src.random_unitary import random_unitary


def run(dm: DM.DensityMatrix, measurement_set, num_iterations: int, Unitaries, orders):
    """

    Args:
        dm: the density matrix to evolve
        measurement_set: a list of functions that take a density matrix as an argument and return a number
        num_iterations: An integer representing the number of iterations the system will go through
        Unitaries: either a list of DMs to be used to evolve the system
                       or: a single unitary to be used at each step
                       or: None, in which case random unitaries will be generated at each step.
        orders: A list of qbit orderes at each iteration step.

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    if type(measurement_set) != list:
        measurement_set = [measurement_set]

    measurement_values = [np.array(measurement(dm)) for measurement in measurement_set]

    generate_random_unitary = False

    if type(Unitaries) == list:
        assert len(Unitaries) == num_iterations, "There must be a unitary for each trial"
    elif type(Unitaries) == DM.DensityMatrix:
        Unitaries = [Unitaries for _ in range(num_iterations)]
    else:
        generate_random_unitary = True
        print("using random unitaries")

    assert len(orders) == num_iterations, "There must be an order for each trial"

    for i in range(num_iterations):

        order = orders[i]

        if generate_random_unitary:
            U = DM.tensor([random_unitary(len(g)) for g in order])
        else:
            U = Unitaries[i]

        U.relabel_basis(order)
        U.change_to_energy_basis()

        dm = U * dm * U.H

        measurement_values = [np.vstack((measurement_values[i], measurement(dm))) for i, measurement in enumerate(measurement_set)]

    return measurement_values


def save_data(data: np.ndarray, num_qbits: str, measurement: str, num_chunks: str, connectivity_type: str, run_index: str, sim_index=int, extra=""):
    if extra != "":
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}_other={extra}_index={sim_index}"
    else:
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}_index={sim_index}"
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = path + f"/{measurement}_{run_index}.dat"
    np.savetxt(file_name, data, header=f"{measurement} for {num_qbits} qbits with connectivity {connectivity_type} in chunks {num_chunks}")
