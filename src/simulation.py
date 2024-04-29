import numpy as np

import src.measurements as measure
from src.setup import xp

import os

import src.density_matrix as DM
from src.random_unitary import random_energy_preserving_unitary


def run(dm: DM.DensityMatrix, num_iterations: int, order_rule, first_order, sub_unitary, connectivity,
        Unitaries=None,

        verbose=False):
    """
    Args:
        dm: the density matrix to evolve
        measurement_set: a list of functions that take a density matrix as an argument and return a number
        num_iterations: An integer representing the number of iterations the system will go through
        num_chunks : The number of subsystems that the full system will be broken in to.
        Unitaries: either a list of DMs to be used to evolve the system, if there are fewer unitaries than iterations they will be used cyclically.
                       or: a single unitary to be used at each step
                       or: None, in which case random unitaries will be generated at each step.
        order_rule: a function that takes (past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary), see example in order_rules.py
        verbose: a float or false. if it is a float between zero and 1 progress will be reported every verbose percent. i.e verbose =.1 will give ten progress reports

    Returns: measurement results and A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """

    pops_values = {0: {index: pop for index, pop in enumerate(measure.pops(dm))}}
    two_qubit_dms = {0: measure.two_qbit_dm_of_every_pair(dm)}
    three_qubit_dms = {0: measure.three_qbit_dm_of_every_triplet(dm)}

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

    for i in range(1, num_iterations):
        if i == 1:
            order = first_order
            previous_order = first_order
            

        chunk_sizes = [len(chunk) for chunk in order]
        leftovers = dm.number_of_qbits % np.sum(chunk_sizes)
        if leftovers:
            leftover_identity = DM.Identity(DM.energy_basis(leftovers))

        progress = i / num_iterations
        if verbose and int(progress * 1000) % int(verbose * 1000) == 0:
            percent = str(int(progress * 100)).zfill(2)
            print(f"{percent}%")

        if generate_random_unitary:

            U = DM.tensor([random_energy_preserving_unitary(chunk_size) for chunk_size in chunk_sizes])

            if leftovers:
                U = U.tensor(leftover_identity)
        else:
            U = Unitaries[i % num_unitaries]

        dm = step(dm, order, U, not generate_random_unitary)
        # Does one qubit measurements on the entire qubit density matrix at each step
        pops_values[i] = {index: pop for index, pop in enumerate(measure.pops(dm))}

        two_qubit_dms[i] = measure.two_qbit_dm_of_every_pair(dm)

        three_qubit_dms[i] = measure.three_qbit_dm_of_every_triplet(dm)

        # the next
        # (past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary):
        order = order_rule(previous_order, pops_values[i - 1], pops_values[i], two_qubit_dms[i - 1], two_qubit_dms[i], connectivity, sub_unitary, dm)

    return (pops_values, two_qubit_dms, three_qubit_dms), dm


def step(dm: DM.DensityMatrix, order: list[np.ndarray], Unitary: DM.DensityMatrix,
         unitary_reused=False) -> DM.DensityMatrix:
    """
    Args:
        dm: the density matrix to evolve
        order: the qbit order to be used e.g. [0,2,1,3]
        Unitary: A Unitary that will be used to evolve the system
        unitary_reused: if the unitary will be reused make sure to undo the reordering

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    # make sure each qbit is assigned to a group and that there are no extras or duplicates.
    # flatten order using a list comprehension
    order = [qbit for chunk in order for qbit in chunk]

    assert set(list(order)) == set(range(dm.number_of_qbits)), f"{set(order)} vs {set(range(dm.number_of_qbits))}"
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


def save_data(data: np.ndarray, num_qbits: str, measurement: str, num_chunks: str, connectivity_type: str,
              run_index: str, sim_index=int, extra=""):
    if extra != "":
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}_other={extra}/index={sim_index}"
    else:
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}/index={sim_index}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    file_name = path + f"/{measurement}_{run_index}.dat"
    np.savetxt(file_name, data,
               header=f"{measurement} for {num_qbits} qbits with connectivity {connectivity_type} in chunks {num_chunks}")
