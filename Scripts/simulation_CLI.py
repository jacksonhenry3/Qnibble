"""
example usage:
python .\simulation_CLI.py --ordering_type c5 --ordering_seed 0  --unitary_seed 0 --unitary_energy_subspace 2 --chunk_size 4 --num_steps 100 --pops .2,.2,.2,.4,.2,.2,.2,.2
"""

# Add directory above current directory to path
import os.path
import sys, argparse

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..')

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary)


def execute(ordering_type, ordering_seed, unitary_energy_subspace, unitary_seed, chunk_size, num_steps, initial_pops, evolution_generation_type="unitary"):


    num_qbits = len(initial_pops)

    assert num_qbits % chunk_size == 0, "Chunk size must divide number of qubits"
    num_chunks = num_qbits // chunk_size
    print("====================================")
    # confirm the argument values
    print(f"chunk size: {chunk_size}")
    print(f"num steps: {num_steps}")
    print(f"initial pops: {initial_pops}")
    print(f"unitary energy subspace: {unitary_energy_subspace}")
    print(f"unitary seed: {unitary_seed}")
    print(f"ordering seed: {ordering_seed}")
    print("====================================")
    unitary_rng = np.random.default_rng(unitary_seed)
    print()
    print(f"generating {ordering_type} ordering")
    match ordering_type:
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
            raise ValueError(f"ordering type {ordering_type} not recognized")

    print("ordering generated\n")
    basis = DM.energy_basis(chunk_size)
    identity = DM.Identity(basis)
    print("generating unitary")
    if unitary_energy_subspace:

        unitary_energy_subspace = int(unitary_energy_subspace)

        # match evolution_generation_type: unitary, hamiltonian, hamiltonian_old
        match evolution_generation_type:
            case "unitary":
                sub_unitary = random_unitary.random_unitary_in_subspace(num_qbits=chunk_size, energy_subspace=unitary_energy_subspace, seed=unitary_rng)
            case "unitary.05":
                sub_unitary = random_unitary.random_unitary_in_subspace(num_qbits=chunk_size, energy_subspace=unitary_energy_subspace, seed=unitary_rng)
                sub_unitary = sub_unitary**.05
            case "hamiltonian":
                sub_hamiltonian = random_unitary.random_hamiltonian_in_subspace(num_qbits=chunk_size, energy_subspace=unitary_energy_subspace, seed=unitary_rng)
                sub_unitary = DM.dm_exp(sub_hamiltonian * -1j * .1)
            case "hamiltonian_old":
                sub_hamiltonian = random_unitary.random_hamiltonian_in_subspace_coppying_mathematica(num_qbits=chunk_size, energy_subspace=unitary_energy_subspace, seed=unitary_rng)
                sub_unitary = DM.dm_exp(sub_hamiltonian * -1j * .1)
            case _:
                # throw an explanatory error
                raise ValueError(f"evolution_generation_type {evolution_generation_type} not recognized")

        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
        unitary = np.prod(composite_unitaries)
    else:
        # match evolution_generation_type: unitary, hamiltonian, hamiltonian_old
        match evolution_generation_type:
            case "unitary":
                sub_unitary = random_unitary.random_energy_preserving_unitary(num_qbits=num_qbits, seed=unitary_rng)
            case "unitary.05":
                sub_unitary = random_unitary.random_energy_preserving_unitary(num_qbits=num_qbits, seed=unitary_rng)
                sub_unitary = sub_unitary**.05
            case "hamiltonian":
                hamiltonian = random_unitary.random_hamiltonian(num_qbits=num_qbits, seed=unitary_rng)
                sub_unitary = DM.dm_exp(hamiltonian * -1j * .1)
            case "hamiltonian_old":
                # throw an incompatible error
                raise ValueError(f"evolution_generation_type {evolution_generation_type} not yet compatible with unitary_energy_subspace")
            case _:
                # throw an explanatory error
                raise ValueError(f"evolution_generation_type {evolution_generation_type} not recognized")

        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
        unitary = np.prod(composite_unitaries)

    print("unitary generated\n")
    print("constructing system")
    system = DM.n_thermal_qbits(initial_pops)
    system.change_to_energy_basis()
    measurements = [measure.pops, measure.extractable_work_of_each_qubit]
    print("running simulation")
    data = sim.run(system,
                   measurement_set=measurements,
                   num_iterations=num_steps,
                   orders=ordering,
                   Unitaries=unitary,
                   verbose=.1
                   )[0]
    path = f"../data/{num_qbits}_{ordering_type}_{unitary_seed}{unitary_energy_subspace}"
    print(f"simulation complete, extracting and saving data to : {path}\n")
    ordering_seed = str(ordering_seed).zfill(3)
    pops = np.array(data[0]).squeeze()
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(f"{path}/pops{ordering_seed}.dat", pops, header=f"pops for {num_qbits} qbits with connectivity {ordering_type} and unitary {unitary}")
    ex_work = np.array(data[1]).squeeze()
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(f"{path}/exwork{ordering_seed}.dat", ex_work, header=f"ex_work for {num_qbits} qbits with connectivity {ordering_type} and unitary {unitary}")
    print("data saved, exiting")


    return (pops, ex_work)


if __name__ == "__main__":
    print("parsing arguments")
    parser = argparse.ArgumentParser(description='This is the CLI to run simulations')

    # Add arguments based on your requirements
    parser.add_argument('--ordering_type', '-o', help='Type of ordering to use [gas,messenger,c5,c6,c7]', default='gas')
    parser.add_argument('--ordering_seed', '-os', type=int, help='the seed for generating the ordering', default=None)
    parser.add_argument('--unitary_energy_subspace', '-ues', type=int, help='(optional) the energy subspace for the subunitary to be in', default=None)
    parser.add_argument('--unitary_seed', '-us', type=int, help='unitary seed', default=None)
    parser.add_argument('--chunk_size', '-cs', type=int, default=4, help='Chunk size')
    parser.add_argument('--num_steps', '-ns', type=int, help='Number of steps')
    parser.add_argument('--pops', '-p', help='Initial populations')

    args = parser.parse_args()

    # Access the parsed arguments
    ordering_type = args.ordering_type
    ordering_seed = args.ordering_seed
    unitary_energy_subspace = args.unitary_energy_subspace
    unitary_seed = args.unitary_seed
    num_steps = args.num_steps
    chunk_size = args.chunk_size
    initial_pops = [float(p) for p in args.pops.split(",")]

    execute(ordering_type, ordering_seed, unitary_energy_subspace, unitary_seed, chunk_size, num_steps, initial_pops)
