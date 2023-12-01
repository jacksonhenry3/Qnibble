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

print("parsing arguments")
parser = argparse.ArgumentParser(description='This is the CLI to run simulations')

# Add arguments based on your requirements
parser.add_argument('--ordering_type', '-o', help='Type of ordering to use [gas,messenger,c5,c6,c7]', default='gas')
parser.add_argument('--ordering_seed', '-os', type=int, help='the seed for generating the ordering', default=None)
parser.add_argument('--unitary_energy_subspace', '-ues', type=int, help='(optional) the energy subspace for the subunitary to be in', default=None)
parser.add_argument('--unitary_seed', '-us', type=int, help='unitary seed', default=None)
parser.add_argument('--unitary_reused', '-ur', type=bool, help='weather to use the same unitary at each step or not', default=False)
parser.add_argument('--chunk_size', '-cs', type=int, default=4, help='Chunk size')
parser.add_argument('--num_steps', '-ns', type=int, help='Number of steps')
parser.add_argument('--pops', '-p', help='Initial populations')

args = parser.parse_args()

# Access the parsed arguments
ordering_type = args.ordering_type
ordering_seed = args.ordering_seed
unitary_energy_subspace = args.unitary_energy_subspace
unitary_seed = args.unitary_seed
unitary_reused = args.unitary_reused
num_steps = args.num_steps
chunk_size = args.chunk_size
initial_pops = [float(p) for p in args.pops.split(",")]
num_qbits = len(initial_pops)

assert num_qbits % chunk_size == 0, "Chunk size must divide number of qubits"
num_chunks = num_qbits // chunk_size

print("====================================")
# confirm the argument values
print(f"chunk size: {chunk_size}")
print(f"num steps: {num_steps}")
print(f"initial pops: {initial_pops}")

print(f"unitary reused: {unitary_reused}")
print(f"unitary energy subspace: {unitary_energy_subspace}")
print(f"unitary seed: {unitary_seed}")
print(f"ordering seed: {ordering_seed}")

print("====================================")

unitary_rng = np.random.default_rng(unitary_seed)
print()
print(f"generating {ordering_type} ordering")

if ordering_type == "gas":
    ordering = orders.n_random_gas_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
elif ordering_type == "c5":
    ordering = orders.n_random_c5_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
elif ordering_type == "c6":
    ordering = orders.n_random_c6_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
elif ordering_type == "c7":
    ordering = orders.n_random_c7_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)
elif ordering_type == "messenger":
    ordering = orders.n_random_messenger_orders(num_qbits=num_qbits, n=num_steps, seed=ordering_seed)

print("ordering generated\n")

basis = DM.energy_basis(chunk_size)
identity = DM.Identity(basis)

print("generating unitary")
if not unitary_reused:
    if unitary_energy_subspace:
        unitary_energy_subspace = int(unitary_energy_subspace)
        sub_unitary = random_unitary.random_unitary_in_subspace(num_qbits=chunk_size, energy_subspace=unitary_energy_subspace, seed=unitary_rng)
        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
        unitary = np.product(composite_unitaries)

    else:
        sub_unitary = random_unitary.random_energy_preserving_unitary(num_qbits=chunk_size, seed=unitary_rng)
        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
        unitary = np.product(composite_unitaries)
else:
    if unitary_energy_subspace.is_digit():
        unitary_energy_subspace = int(unitary_energy_subspace)
        unitary = []
        for _ in range(num_steps):
            sub_unitary = random_unitary.random_unitary_in_subspace(num_qbits=chunk_size, energy_subspace=unitary_energy_subspace, seed=unitary_rng)
            composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
            a_unitary = np.product(composite_unitaries)
            unitary.append(a_unitary)
    else:
        unitary = []
        for _ in range(num_steps):
            sub_unitary = random_unitary.random_energy_preserving_unitary(num_qbits=chunk_size, seed=unitary_rng)
            composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
            a_unitary = np.product(composite_unitaries)
            unitary.append(a_unitary)

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

path = f"../data/{num_qbits}_{ordering_type}_{unitary_seed}{unitary_reused}{unitary_energy_subspace}"
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
