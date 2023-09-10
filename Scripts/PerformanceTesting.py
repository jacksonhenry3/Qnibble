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

num_iterations = 50
measurements = []
num_qbits = 12
system = DM.n_thermal_qbits([random.random()/2. for _ in range(num_qbits)])


ordering = orders.n_random_line_orders(chunk_sizes=[4 for _ in range(num_qbits // 4)], n=100, num_qbits=num_qbits)

data, result = sim.run(system,
                       measurement_set=measurements,
                       num_iterations=num_iterations,
                       orders=ordering,
                       verbose=.001,
                       )

# result.plot()
# plt.show()
