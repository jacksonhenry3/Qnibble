# Add directory above current directory to path
import os.path
import sys, argparse, pickle

sys.path.insert(0, '..')

import numpy as np

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary)

print("parsing arguments")
parser = argparse.ArgumentParser(description='This is the CLI to run simulations')

# Add arguments based on your requirements
parser.add_argument('--ordering_path', '-o', help='Path to the connectivity file')
parser.add_argument('--unitary_path', '-u', help='Path to the unitary file')
parser.add_argument('--num_steps', '-n', type=int, help='Number of steps')
parser.add_argument('--index', '-i', type=int, help='Index value')

# Add more arguments as needed

# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
ordering_path = args.ordering_path
unitary_path = args.unitary_path
num_steps = args.num_steps
index = args.index

# load the connectivity using numpy
ordering = np.load(ordering_path)

# load the unitary using pickle
with open(unitary_path, 'rb') as file:
    unitary = pickle.load(file)

# get the number of qubits from the connectivity
num_qbits = ordering.shape[2] * ordering.shape[1]

initial_pops = [.2 for _ in range(num_qbits)]
initial_pops[0] = .4

system = DM.n_thermal_qbits(initial_pops)
system.change_to_energy_basis()

measurements = [measure.pops, measure.extractable_work_of_each_qubit]

data = sim.run(system,
               measurement_set=measurements,
               num_iterations=num_steps,
               orders=ordering,
               Unitaries=unitary,
               verbose=.1
               )[0]

pops = np.array(data[0]).squeeze()

order = os.path.splitext(os.path.basename(ordering_path))[0][:2]
unitary = os.path.splitext(os.path.basename(unitary_path))[0]

index = str(index).zfill(3)
path = f"../data/{num_qbits}_{order}_{unitary}"
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(f"{path}/pops{index}.dat", pops, header=f"pops for {num_qbits} qbits with connectivity {order} and unitary {unitary}")

ex_work = np.array(data[1]).squeeze()
path = f"../data/{num_qbits}_{order}_{unitary}"
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(f"{path}/exwork{index}.dat", ex_work, header=f"ex_work for {num_qbits} qbits with connectivity {order} and unitary {unitary}")