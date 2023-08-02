# Add directory above current directory to path
import sys as SYS;

SYS.path.insert(0, '..')
import os
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

bath_size = 4
hot_pop = .49
cold_pop = .01

system_size = 4
system_pop = .25

pops = [system_pop for _ in range(system_size)] + [hot_pop for _ in range(bath_size)] + [cold_pop for _ in range(bath_size)]
system = DM.n_thermal_qbits(pops)

interacting_with_the_hot_bath = [np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9, 10, 11])]
interacting_with_the_cold_bath = [np.array([0, 1, 2, 3, 8, 9, 10, 11]), np.array([4, 5, 6, 7])]
interacting_with_the_self = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])]

orders = [interacting_with_the_hot_bath for _ in range(5)]
orders += [interacting_with_the_self for _ in range(5)]
num_iterations = 5

measurments = [measure.pops, measure.extractable_work_of_each_qubit]
results = sim.run(system,
                  measurement_set=measurments,
                  num_iterations=num_iterations,
                  orders=orders,
                  qbits_to_measure=[0, 1, 2, 3],
                  verbose=True
                  )[0];
