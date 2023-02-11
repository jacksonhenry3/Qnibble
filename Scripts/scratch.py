from matplotlib import pyplot as plt
# from scalene import scalene_profiler



# Add directory above current directory to path
import sys as SYS

from measurements import mutual_information
from random_unitary import random_energy_preserving_unitary
from simulation import step

SYS.path.insert(0, '..')
SYS.path.insert(0, '')
import os

from src import setup

setup.use_gpu()
from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders)

N = 8
num_chunks = 2
num_iterations = 100


ordering = orders.n_random_line_orders(line_length=N, n=num_iterations)
initial_pops = [.2 for _ in range(N)]
initial_pops[5] = .4

system = DM.n_thermal_qbits(initial_pops)
system.change_to_energy_basis()

MI = []
for iteration_index in range(num_iterations):
    U = DM.tensor([random_energy_preserving_unitary(N//num_chunks) for _ in range(num_chunks)])
    system = step(system,[0,1,2,3,4,5,6,7],U)
    MI.append(mutual_information(system,[0,1,2,3]))
plt.plot(MI)
plt.show()
