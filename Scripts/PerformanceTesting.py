# Add directory above current directory to path
import sys as SYS;

SYS.path.insert(0, '..')
import random
# for saving
import os

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from src import setup

# setup.use_gpu()

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary,
    simulation)

num_iterations = 150
measurements = [measure.extractable_work_of_each_qubit]
num_qbits = 16
pops = [.2 for _ in range(num_qbits)]
pops[0] = .4
system = DM.n_thermal_qbits(pops)

ordering = orders.n_random_line_orders(chunk_sizes=[4 for _ in range(num_qbits // 4)], n=100, num_qbits=num_qbits)

data, result = sim.run(system,
                       measurement_set=measurements,
                       num_iterations=num_iterations,
                       orders=ordering,
                       verbose=.001,
                       )

for i,datum in enumerate(data):
    print(i)
    sim.save_data(data=datum, connectivity_type="line", run_index=str(0), sim_index=i, num_qbits=str(num_qbits), num_chunks=str(num_qbits // 4), measurement="extractable_work_of_each_qubit")
