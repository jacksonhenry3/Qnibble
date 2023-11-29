# Add directory above current directory to path
import sys

sys.path.insert(0, '..')

import numpy as np

from src import setup
# setup.use_gpu()

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary)

_, N, num_iterations, order, index = sys.argv
N = int(N)
num_iterations = int(num_iterations)

chunk_size = 4
num_chunks = N // chunk_size

identity = DM.Identity(DM.energy_basis(chunk_size))

initial_pops = [.2 for _ in range(N)]
initial_pops[4] = .4

measurements = [measure.pops, measure.extractable_work_of_each_qubit]

gas_orderings = orders.n_random_gas_orders(num_qbits=N, chunk_sizes=[chunk_size for _ in range(num_chunks)], n=num_iterations)
c6_orderings = orders.n_random_c6_orders(num_qbits=N, chunk_sizes=[chunk_size for _ in range(num_chunks)], n=num_iterations)
c5_orderings = orders.n_random_c5_orders(num_qbits=N, n=num_iterations)
c7_orderings = orders.n_random_c7_orders(num_qbits=N, n=num_iterations)
messenger_orderings = orders.n_alternating_messenger_orders(num_qbits=N, n=num_iterations)

ordering = gas_orderings if order == "gas" else c5_orderings if order == "5" else c6_orderings if order == "6" else c7_orderings if order == "7" else messenger_orderings

sub_unitary = random_unitary.random_unitary_in_subspace(4, 2)
composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
unitary = np.product(composite_unitaries)

system = DM.n_thermal_qbits(initial_pops)
system.change_to_energy_basis()

data = sim.run(system,
               measurement_set=measurements,
               num_iterations=num_iterations,
               orders=ordering,
               Unitaries=unitary,
               verbose=.1
               )[0]

sim.save_data(np.array(data[0]).squeeze(), str(N), "pops", str(num_chunks), order, run_index=str(index), sim_index=str(index), extra="")
sim.save_data(np.array(data[1]).squeeze(), str(N), "ex_work", str(num_chunks), order, run_index=str(index), sim_index=str(index), extra="")
