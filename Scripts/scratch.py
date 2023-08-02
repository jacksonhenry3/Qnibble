import random

import numpy as np
from matplotlib import pyplot as plt
# from scalene import scalene_profiler


# Add directory above current directory to path
import sys as SYS

import simulation

SYS.path.insert(0, '..')
SYS.path.insert(0, '')
import os

from src import setup

# setup.use_gpu()
from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    random_unitary,
    orders)

# test out the mask
DM._ptrace_mask(14,(0,1))
# N = 8
# num_chunks = 4
# num_iterations = 20
# chunk_size = N // num_chunks
#
# ordering = orders.n_random_line_orders(line_length=N, n=num_iterations)
#
#
# initial_pops = [random.random() for _ in range(N)]
# system = DM.n_thermal_qbits(initial_pops)
# system.change_to_energy_basis()
#
# _,system = simulation.run(system, num_iterations=num_iterations, orders=ordering, verbose=False, num_chunks=3, measurement_set=[])
# # system.change_to_canonical_basis()
# system.plot()
# plt.show()
