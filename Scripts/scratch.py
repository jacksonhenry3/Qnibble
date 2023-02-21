import random

import numpy as np
from matplotlib import pyplot as plt
# from scalene import scalene_profiler


# Add directory above current directory to path
import sys as SYS

SYS.path.insert(0, '..')
SYS.path.insert(0, '')
import os

from src import setup

setup.use_gpu()
from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    random_unitary,
    orders)

N = 8
num_chunks = 2
num_iterations = 20
chunk_size = N // num_chunks

ordering = orders.n_random_line_orders(line_length=N, n=num_iterations)
initial_pops = [random.random() for _ in range(N)]
initial_pops[2] = .3

system = DM.n_thermal_qbits(initial_pops)
system.change_to_energy_basis()

sub_system_qbits = [0, 1, 2]
environment_qbits = list(set(range(system.basis.num_qubits)) - set(sub_system_qbits))
print(environment_qbits)
S1 = []
S2 = []
Stot = []
sub_system = system.ptrace(environment_qbits)
environment = system.ptrace(sub_system_qbits)
S1.append(measure.entropy(sub_system))
S2.append(measure.entropy(environment))
Stot.append(measure.entropy(system))

MI = []
MI.append(measure.mutual_information(system, sub_system_qbits))
for iteration_index in range(num_iterations):
    print(iteration_index)
    U = DM.tensor([random_unitary.random_energy_preserving_unitary(chunk_size) for i in range(num_chunks)])
    system = sim.step(system, ordering[iteration_index], U)


    mi = measure.mutual_information(system, sub_system_qbits)
    MI.append(mi)

plt.plot(S1, label="system")
plt.plot(S2, label="environment")
plt.plot(Stot, label="Total")
plt.plot(MI, label="Mutual Information")
plt.legend()
plt.show()
