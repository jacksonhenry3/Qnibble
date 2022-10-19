# Add directory above current directory to path
import sys as SYS;

SYS.path.insert(0, '..')
import os

from src import setup

setup.use_gpu()

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders)

N = 14
num_chunks = 7
num_iterations = 50
measurments = []
ordering = orders.n_random_line_orders(line_length=N, n=num_iterations)
initial_pops = [.2 for _ in range(N)]
initial_pops[5] = .4

system = DM.n_thermal_qbits(initial_pops)
system.change_to_energy_basis()

results = sim.run(system,
                  measurement_set=measurments,
                  num_iterations=num_iterations,
                  num_chunks=num_chunks,
                  orders=ordering,
                  verbose=.001)
#
# sim.save_data(results[0], N, "pops", num_chunks, "line", 0, 0)
# sim.save_data(results[1], N, "extractable_work", num_chunks, "line", 0, 0)
