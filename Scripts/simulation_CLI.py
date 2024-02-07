"""
example usage:
python .\simulation_CLI.py --output_file_name an_example_run --ordering_type c5 --ordering_seed 0  --unitary_seed 0 --unitary_energy_subspace 2 --chunk_size 4 --num_steps 100 --pops .2,.2,.2,.4,.2,.2,.2,.2
"""

# Add directory above current directory to path
import os.path
import sys, argparse, h5py

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..')

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary)


def execute(file_name: str, connectivity, ordering_seed, unitary_energy_subspace, unitary_seed, num_steps, initial_pops, chunk_size=4,
            evolution_generation_type="unitary", verbosity=.1, sample_frequency=5):
    """
    file_name: name of the file to save the data to (without the .hdf5 extension) example: "ZestyGodzilla"
    connectivity: the type of connectivity to use for the ordering. options: "gas", "c5", "c6", "c7", ("messenger" has some issues)
    ordering_seed: the seed to use for the ordering
    unitary_energy_subspace: the energy subspace to use for the unitary evolution
    unitary_seed: the seed to use for the unitary evolution
    num_steps: the number of steps to take
    initial_pops: the initial populations of the qubits
    chunk_size: the size of the chunks to use for the unitary evolution
    evolution_generation_type: the type of evolution to use. options: "unitary","unitary.05","hamiltonian", "hamiltonian_old", for both hamiltonians the dtheta is .1
    verbosity: the verbosity of progress reports. .1 is every 10%, .01 is every 1%, etc.
    """

    num_qbits = len(initial_pops)

    assert num_qbits % chunk_size == 0, "Chunk size must divide number of qubits"
    num_chunks = num_qbits // chunk_size

    # the if __name__ == "__main__": statements make the statements only run if this file is run directly, not if it is imported.
    if __name__ == "__main__": print("====================================")
    # confirm the argument values
    if __name__ == "__main__": print(f"chunk size: {chunk_size}")
    if __name__ == "__main__": print(f"num steps: {num_steps}")
    if __name__ == "__main__": print(f"initial pops: {initial_pops}")
    if __name__ == "__main__": print(f"unitary energy subspace: {unitary_energy_subspace}")
    if __name__ == "__main__": print(f"unitary seed: {unitary_seed}")
    if __name__ == "__main__": print(f"ordering seed: {ordering_seed}")
    if __name__ == "__main__": print("====================================")
    unitary_rng = np.random.default_rng(unitary_seed)
    if __name__ == "__main__": print()
    if __name__ == "__main__": print(f"generating {connectivity} ordering")
    match connectivity:
        case "gas":
            ordering = orders.n_random_gas_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
        case "c5":
            ordering = orders.n_random_c5_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
        case "c6":
            ordering = orders.n_random_c6_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
        case "c7":
            ordering = orders.n_random_c7_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
        case "messenger":
            ordering = orders.n_random_messenger_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
        case _:
            # throw an explanatory error
            raise ValueError(f"ordering type {connectivity} not recognized")

    if __name__ == "__main__": print("ordering generated\n")
    basis = DM.energy_basis(chunk_size)
    identity = DM.Identity(basis)
    if __name__ == "__main__": print("generating unitary")
    if unitary_energy_subspace:

        unitary_energy_subspace = int(unitary_energy_subspace)

        # match evolution_generation_type: unitary, hamiltonian, hamiltonian_old
        match evolution_generation_type:
            case "unitary":
                sub_unitary = random_unitary.random_unitary_in_subspace(num_qbits=chunk_size,
                                                                        energy_subspace=unitary_energy_subspace,
                                                                        seed=unitary_rng)
            case "unitary.05":
                sub_unitary = random_unitary.random_unitary_in_subspace(num_qbits=chunk_size,
                                                                        energy_subspace=unitary_energy_subspace,
                                                                        seed=unitary_rng)
                sub_unitary = sub_unitary ** .05
            case "hamiltonian":
                sub_hamiltonian = random_unitary.random_hamiltonian_in_subspace(num_qbits=chunk_size,
                                                                                energy_subspace=unitary_energy_subspace,
                                                                                seed=unitary_rng)
                sub_unitary = DM.dm_exp(sub_hamiltonian * -1j * .1)
            case "hamiltonian_old":
                sub_hamiltonian = random_unitary.random_hamiltonian_in_subspace_coppying_mathematica(
                    num_qbits=chunk_size, energy_subspace=unitary_energy_subspace, seed=unitary_rng)
                sub_unitary = DM.dm_exp(sub_hamiltonian * -1j * .1)
            case _:
                # throw an explanatory error
                raise ValueError(f"evolution_generation_type {evolution_generation_type} not recognized")

        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in
                               range(num_chunks)]
        unitary = np.prod(composite_unitaries)
    else:
        # match evolution_generation_type: unitary, hamiltonian, hamiltonian_old
        match evolution_generation_type:
            case "unitary":
                sub_unitary = random_unitary.random_energy_preserving_unitary(num_qbits=chunk_size, seed=unitary_rng)
            case "unitary.05":
                sub_unitary = random_unitary.random_energy_preserving_unitary(num_qbits=chunk_size, seed=unitary_rng)
                sub_unitary = sub_unitary ** .05
            case "hamiltonian":
                hamiltonian = random_unitary.random_hamiltonian(num_qbits=chunk_size, seed=unitary_rng)
                sub_unitary = DM.dm_exp(hamiltonian * -1j * .1)
            case "hamiltonian_old":
                # throw an incompatible error
                raise ValueError(
                    f"evolution_generation_type {evolution_generation_type} not yet compatible with unitary_energy_subspace")
            case _:
                # throw an explanatory error
                raise ValueError(f"evolution_generation_type {evolution_generation_type} not recognized")

        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in
                               range(num_chunks)]
        unitary = np.prod(composite_unitaries)

    if __name__ == "__main__": print("unitary generated\n")
    if __name__ == "__main__": print("constructing system")
    system = DM.n_thermal_qbits(initial_pops)
    system.change_to_energy_basis()
    measurements = [measure.two_qbit_dm_of_every_pair, measure.three_qbit_dm_of_every_triplet]
    # measurements = [measure.pops, measure.extractable_work_of_each_qubit, measure.mutual_information_of_every_pair]
    if __name__ == "__main__": print("running simulation")

    pops, two_qubit_dms = sim.run(system,
                                  num_iterations=num_steps,
                                  orders=ordering,
                                  sample_frequency=sample_frequency,
                                  Unitaries=unitary,
                                  verbose=verbosity,
                                  )[0]




    save_data(file_name=file_name, data=two_qubit_dms, connectivity=connectivity, unitary_energy_subspace=unitary_energy_subspace, unitary_seed=unitary_seed, ordering_seed=ordering_seed,
              measurment="two_qubit_dms", num_qubits=num_qbits)
    save_data(file_name=file_name, data=pops, connectivity=connectivity, unitary_energy_subspace=unitary_energy_subspace, unitary_seed=unitary_seed, ordering_seed=ordering_seed,
              measurment="pops", num_qubits=num_qbits)
    if __name__ == "__main__": print("data saved, exiting")
    return pops, two_qubit_dms


def save_data(file_name: str, data, connectivity, unitary_energy_subspace, unitary_seed, ordering_seed, measurment, num_qubits):
    path_to_data = os.path.relpath('data')

    while not os.path.isdir(path_to_data):
        path_to_data = os.path.join('..', path_to_data)


    os.makedirs(f"{path_to_data}/{file_name}", exist_ok=True)
    file_name = os.path.join(path_to_data, file_name, f"{file_name}-{num_qubits}_qubits-{connectivity}_connectivity-unitary_energy_subspace_{unitary_energy_subspace}-unitary_seed_{unitary_seed}-ordering_seed_{ordering_seed}")

    print(f"simulation complete, extracting and saving data to : {file_name}")


    group_name = f"{num_qubits} qubits/{connectivity} connectivity/unitary energy subspace {unitary_energy_subspace}/unitary seed {unitary_seed}/ordering seed {ordering_seed}/{measurment}"

    file = h5py.File(file_name + ".hdf5", "a")
    if group_name not in file:
        file.create_group(group_name)
    group = file[group_name]

    for time_index in data:
        # check if the group already exists
        group_name = f'{time_index}'
        sub_index = 0
        while group_name in group:
            group_name = f'{time_index}({sub_index})'
            sub_index += 1

        time_slice = group.create_group(group_name)
        for key, value in data[time_index].items():

            # if the value is a scalar
            if np.isscalar(value):
                time_slice.create_dataset(str(key), data=value)
            else:
                time_slice.create_dataset(str(key), data=value.data.toarray())

    file.close()


if __name__ == "__main__":
    print("parsing arguments")
    parser = argparse.ArgumentParser(description='This is the CLI to run simulations')

    # Add arguments based on your requirements
    parser.add_argument('--output_file_name', '-f', help='Name of the output file')
    parser.add_argument('--ordering_type', '-o', help='Type of ordering to use [gas,messenger,c5,c6,c7]', default='gas')
    parser.add_argument('--ordering_seed', '-os', type=int, help='the seed for generating the ordering', default=None)
    parser.add_argument('--unitary_energy_subspace', '-ues', type=int,
                        help='(optional) the energy subspace for the subunitary to be in', default=None)
    parser.add_argument('--unitary_seed', '-us', type=int, help='unitary seed', default=None)
    parser.add_argument('--chunk_size', '-cs', type=int, default=4, help='Chunk size')
    parser.add_argument('--num_steps', '-ns', type=int, help='Number of steps')
    parser.add_argument('--pops', '-p', help='Initial populations')

    args = parser.parse_args()

    # Access the parsed arguments
    file_name = args.output_file_name
    ordering_type = args.ordering_type
    ordering_seed = args.ordering_seed
    unitary_energy_subspace = args.unitary_energy_subspace
    unitary_seed = args.unitary_seed
    num_steps = args.num_steps
    chunk_size = args.chunk_size
    initial_pops = [float(p) for p in args.pops.split(",")]

    execute(file_name=file_name,
            connectivity=ordering_type,
            ordering_seed=ordering_seed,
            unitary_energy_subspace=unitary_energy_subspace,
            unitary_seed=unitary_seed,
            num_steps=num_steps,
            chunk_size=chunk_size,
            initial_pops=initial_pops)
