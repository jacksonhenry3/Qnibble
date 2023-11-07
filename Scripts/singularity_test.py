# Add directory above current directory to path
import sys as SYS;

SYS.path.insert(0, '..')

import numpy as np

# from src import setup
# setup.use_gpu()

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary)

N = 16
chunk_size = 4
num_chunks = N // chunk_size

identity = DM.Identity(DM.energy_basis(chunk_size))

num_iterations = 50
num_samples = 10

initial_pops = [.2 for _ in range(N)]
initial_pops[4] = .4

measurements = [measure.pops, measure.extractable_work_of_each_qubit]
gas_orderings = orders.n_random_gas_orders(num_qbits=N, chunk_sizes=[chunk_size for _ in range(num_chunks)], n=num_iterations)
line_orderings = orders.n_random_line_orders(num_qbits=N, chunk_sizes=[chunk_size for _ in range(num_chunks)], n=num_iterations)
messenger_orderings = orders.n_alternating_messenger_orders(num_qbits=N, n=num_iterations)
orderings = [gas_orderings, line_orderings, messenger_orderings]
titles = ["gas", "line", "messenger"]

for i, ordering in enumerate(orderings):
    print(f"starting {titles[i]} with order {ordering}")

    for index in range(num_samples):
        sub_unitary = random_unitary.random_unitary_in_subspace(4, 2)
        unitary = sub_unitary.tensor(identity).tensor(identity).tensor(identity) * \
                  identity.tensor(sub_unitary).tensor(identity).tensor(identity) * \
                  identity.tensor(identity).tensor(sub_unitary).tensor(identity) * \
                  identity.tensor(identity).tensor(identity).tensor(sub_unitary)

        system = DM.n_thermal_qbits(initial_pops)
        system.change_to_energy_basis()

        data = sim.run(system,
                       measurement_set=measurements,
                       num_iterations=num_iterations,
                       orders=ordering,
                       Unitaries=unitary
                       )[0]

        if index % 1 == 0: print(index)
        sim.save_data(np.array(data[0]).squeeze(), str(N), "pops", str(num_chunks), titles[i], run_index=str(index), sim_index=str(index), extra="")
        sim.save_data(np.array(data[1]).squeeze(), str(N), "ex_work", str(num_chunks), titles[i], run_index=str(index), sim_index=str(index), extra="")
