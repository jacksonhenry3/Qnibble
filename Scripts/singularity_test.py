# Add directory above current directory to path
import sys as SYS;

SYS.path.insert(0, '..')
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from src import setup

setup.use_gpu()

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary,
    simulation)

identity = DM.Identity(DM.energy_basis(6))

N = 12
num_chunks = 2
num_iterations = 50
measurments = [measure.pops, measure.extractable_work_of_each_qubit]

initial_pops = [.2 for _ in range(N)]
initial_pops[4] = .4

num_samples = 10

gas_samples_extractable_work = []
gas_sample_pops = []
gas_orderings = orders.n_random_gas_orders(num_qbits=N, chunk_sizes=[6, 6], n=num_iterations)
line_orderings = orders.n_random_line_orders(num_qbits=N, chunk_sizes=[6, 6], n=num_iterations)
messenger_orderings = orders.n_alternating_messenger_orders(num_qbits=N, n=num_iterations)
orderings = [gas_orderings, line_orderings, messenger_orderings]
titles = ["seven", "six", "messenger"]
results = defaultdict(lambda: defaultdict(list))
for i, ordering in enumerate(orderings):
    results[titles[i]]["pops"] = []
    results[titles[i]]["ex_work"] = []
    for index in range(num_samples):
        sub_unitary = random_unitary.random_unitary_in_subspace(6, 2)
        unitary = sub_unitary.tensor(identity) * identity.tensor(sub_unitary)

        system = DM.n_thermal_qbits(initial_pops)
        system.change_to_energy_basis()

        data = sim.run(system,
                       measurement_set=measurments,
                       num_iterations=num_iterations,
                       orders=ordering,
                       Unitaries=unitary
                       )[0]

        if index % 1 == 0: print(index)
        results[titles[i]]["pops"].append(data[0])
        results[titles[i]]["ex_work"].append(data[1])
