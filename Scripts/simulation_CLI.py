"""
example usage:
"""

# Add directory above current directory to path
import os.path
import sys, argparse, h5py

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..')

from src import (
    density_matrix as DM,
    simulation as sim,
    disordered_networks,
    orders,
    random_unitary,
    order_rules)


def execute(file_name: str, connectivity, order_rule_name: str, unitary_energy_subspace, unitary_seed, num_steps, initial_pops,
            evolution_generator_type: str, chunk_size, verbosity=.1, first_10_order=None):
    """
    file_name: name of the file to save the data to (without the .hdf5 extension) example: "ZestyGodzilla"
    connectivity: the type of connectivity to use for the ordering. options: "gas", "c5", "c6", "c7"
    order_rule_name: a string represneting which order rule to use
    unitary_energy_subspace: the energy subspace to use for the unitary evolution
    unitary_seed: the seed to use for the unitary evolution
    num_steps: the number of steps to take
    initial_pops: the initial populations of the qubits
    chunk_size: the size of the chunks to use for the unitary evolution
    evolution_generator_type: the type of evolution to use. options: "unitary","unitary.05","hamiltonian", "hamiltonian_old", for both hamiltonians the dtheta is .1
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
    if __name__ == "__main__": print("====================================")

    unitary_rng = np.random.default_rng(unitary_seed)

    if __name__ == "__main__": print()
    if __name__ == "__main__": print(f"generating {connectivity} ordering")

    match order_rule_name:
        case 'random':
            order_rule = order_rules.random
        case 'greedy':
            order_rule = order_rules.greedy
        case 'strongest_maximizes':
            order_rule = order_rules.strongest_maximizes
        case 'landscape_maximizes':
            order_rule = order_rules.landscape_maximizes
        case 'mimic':
            order_rule = order_rules.mimic
        case 'disorder_random':
            order_rule = disordered_networks.disorder_random
        case 'disorder_greedy_therm':
            order_rule = disordered_networks.disorder_greedy_therm
        case 'disorder_mimic_therm':
            order_rule = disordered_networks.disorder_mimic_therm
        case 'disorder_greedy_v1':
            order_rule = disordered_networks.disorder_greedy_v1
        case 'disorder_mimic_v1':
            order_rule = disordered_networks.disorder_mimic_v1
        case 'disorder_landscape_maximizes':
            order_rule = disordered_networks.disorder_landscape_maximizes
        case'disorder_strongest_maximizes':
            order_rule = disordered_networks.disorder_strongest_maximizes
        case _:
            raise ValueError(f"order_rule_name {order_rule_name} not recognized")


    if first_10_order is None:
        if __name__ == "__main__": print("generating first order")
        match connectivity:
            case 'c2_2local':
                first_10_order = orders.first_10_orders_CN_2local(num_qbits)
            case 'c4_2local':
                first_10_order = orders.first_10_orders_CN_2local(num_qbits)
            #case 'c5':
                #first_order = orders.n_random_c5_orders(num_qbits=num_qbits, chunk_size=chunk_size, n=1, seed=unitary_rng)[0]
            case 'c5_2local':
                first_10_order = orders.first_10_orders_CN_2local(num_qbits)
            case 'c6_2local':
                first_10_order = orders.first_10_orders_CN_2local(num_qbits)
            case 'cN_2local':
                first_10_order = orders.first_10_orders_CN_2local(num_qbits)
            case 'c7':
                first_10_order = orders.n_random_c7_orders(num_qbits=num_qbits, chunk_size=chunk_size, n=10, seed=unitary_rng)
            #case 'gas':
                #first_order = orders.n_random_gas_orders(num_qbits=num_qbits, chunk_size=chunk_size, n=1, seed=unitary_rng)[0]
            #case _:
                # throw an explanatory error
               # raise ValueError(f"connectivity {connectivity} not recognized")

    basis = DM.energy_basis(chunk_size)
    identity = DM.Identity(basis)
    if __name__ == "__main__": print("generating unitary")
    if unitary_energy_subspace:

        unitary_energy_subspace = int(unitary_energy_subspace)

        # match evolution_generator_type: unitary, hamiltonian, hamiltonian_old
        match evolution_generator_type:

            case "haar2Qunitary":
                sub_unitary = random_unitary.haar_random_unitary(theta_divisor=1,phi_divisor=1,omega_divisor=1, seed=None)

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
                raise ValueError(f"evolution_generator_type {evolution_generator_type} not recognized")

        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in
                               range(num_chunks)]
        unitary = np.prod(composite_unitaries)
    else:
        # match evolution_generator_type: unitary, hamiltonian, hamiltonian_old
        match evolution_generator_type:
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
                    f"evolution_generator_type {evolution_generator_type} not yet compatible with unitary_energy_subspace")
            case _:
                # throw an explanatory error
                raise ValueError(f"evolution_generator_type {evolution_generator_type} not recognized")

        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in
                               range(num_chunks)]
        unitary = np.prod(composite_unitaries)

    if __name__ == "__main__": print("unitary generated\n")
    if __name__ == "__main__": print("constructing system")
    system = DM.n_thermal_qbits(initial_pops)
    system.change_to_energy_basis()
    if __name__ == "__main__": print("running simulation")
#three_qubit_dms
#two_qubit_dms

    pops, two_qubit_dms,orders_list = sim.run(system,
                                  num_iterations=num_steps,
                                  Unitaries=unitary,
                                  sub_unitary=sub_unitary,
                                  verbose=verbosity,
                                  order_rule=order_rule,
                                  first_10_order=first_10_order,
                                  connectivity=connectivity,
                                  )[0]
    #save_data(file_name=file_name, data=three_qubit_dms, connectivity=connectivity, unitary_energy_subspace=unitary_energy_subspace, unitary_seed=unitary_seed,
             # order_rule_name=order_rule_name,measurment="three_qubit_dms", num_qubits=num_qbits)
    save_data(file_name=file_name, data=orders_list, connectivity=connectivity,
              unitary_energy_subspace=unitary_energy_subspace, unitary_seed=unitary_seed,
              order_rule_name=order_rule_name,
              measurment="previous_order", num_qubits=num_qbits)
    save_data(file_name=file_name, data=two_qubit_dms, connectivity=connectivity, unitary_energy_subspace=unitary_energy_subspace, unitary_seed=unitary_seed, order_rule_name=order_rule_name,
              measurment="two_qubit_dms", num_qubits=num_qbits)
    save_data(file_name=file_name, data=pops, connectivity=connectivity, unitary_energy_subspace=unitary_energy_subspace, unitary_seed=unitary_seed, order_rule_name=order_rule_name,
              measurment="pops", num_qubits=num_qbits)
    if __name__ == "__main__": print("data saved, exiting")
    return pops, two_qubit_dms,orders_list
#, two_qubit_dms
    #three_qubit_dms


def save_data(file_name: str, data, connectivity, unitary_energy_subspace, unitary_seed, order_rule_name: str, measurment, num_qubits):
    path_to_data = os.path.relpath('data')

    while not os.path.isdir(path_to_data):
        path_to_data = os.path.join('..', path_to_data)

    os.makedirs(f"{path_to_data}/{file_name}", exist_ok=True)
    file_name = os.path.join(path_to_data, file_name,
                             f"{file_name}-{num_qubits}_qubits-{connectivity}_connectivity-unitary_energy_subspace_{unitary_energy_subspace}-unitary_seed_{unitary_seed}-order_rule_name_{order_rule_name}")

    print(f"simulation complete, extracting and saving data to : {file_name}")

    group_name = f"{num_qubits} qubits/{connectivity} connectivity/unitary energy subspace {unitary_energy_subspace}/unitary seed {unitary_seed}/ordering seed {order_rule_name}/{measurment}"

    file = h5py.File(file_name + ".hdf5", "a")
    if group_name not in file:
        file.create_group(group_name)
    group = file[group_name]

    # Handle saving the data
    if isinstance(data, dict):
        for time_index in data:
            # check if the group already exists
            group_name = f'{time_index}'
            sub_index = 0
            while group_name in group:
                group_name = f'{time_index}({sub_index})'
                sub_index += 1

            time_slice = group.create_group(group_name)

            for key, value in data[time_index].items():
                if np.isscalar(value):
                    time_slice.create_dataset(str(key), data=value)
                else:
                    time_slice.create_dataset(str(key), data=value.data.toarray())

    elif isinstance(data, np.ndarray):
        # Handle saving a NumPy array
        data = np.array(data)  # Ensure data is a NumPy array
        time_slice = group.create_group('array_data')  # Create a group for the array data
        time_slice.create_dataset("data", data=data)

    elif isinstance(data, list):
        # Handle saving a list (or 1D array)
        data = np.array(data)
        time_slice = group.create_group('orders_list')
        time_slice.create_dataset("data", data=data)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    file.close()



if __name__ == "__main__":
    print("parsing arguments")
    parser = argparse.ArgumentParser(description='This is the CLI to run simulations')

    # Add arguments based on your requirements
    parser.add_argument('--file_name', '-f', help='Name of the output file')
    parser.add_argument('--connectivity', '-o', help='Type of ordering to use [gas,messenger,c5,c6,c7]', default='gas')
    parser.add_argument('--order_rule_name', '-os', type=str, help='the rule generating the ordering', default=None)
    parser.add_argument('--unitary_energy_subspace', '-ues', type=int,
                        help='(optional) the energy subspace for the subunitary to be in', default=None)
    parser.add_argument('--unitary_seed', '-us', type=int, help='unitary seed', default=None)
    parser.add_argument('--chunk_size', '-cs', type=int, default=2, help='Chunk size')
    parser.add_argument('--num_steps', '-ns', type=int, help='Number of steps')
    parser.add_argument('--initial_pops', '-p', help='Initial populations')
    parser.add_argument('--evolution_generator_type', '-egt', help='Evolution Generator type',default = 'haar2Qunitary')


    args = parser.parse_args()

    # Access the parsed arguments
    file_name = args.file_name
    connectivity = args.connectivity
    order_rule_name = args.order_rule_name
    unitary_energy_subspace = args.unitary_energy_subspace
    unitary_seed = args.unitary_seed
    num_steps = args.num_steps
    #chunk_size = args.chunk_size
    evolution_generator_type = args.evolution_generator_type
    initial_pops = [float(p) for p in args.initial_pops.split(",")]

    execute(file_name=file_name,
            connectivity=connectivity,
            order_rule_name=order_rule_name,
            unitary_energy_subspace=unitary_energy_subspace,
            unitary_seed=unitary_seed,
            num_steps=num_steps,
            initial_pops=initial_pops,
            evolution_generator_type = evolution_generator_type, chunk_size=2)
