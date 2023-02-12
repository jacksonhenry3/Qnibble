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


ordering = orders.n_random_line_orders(line_length=N, n=num_iterations)
initial_pops = [random.random() for _ in range(N)]
initial_pops[2] = .3

system = DM.n_thermal_qbits(initial_pops)
system.change_to_energy_basis()

sub_system_qbits = [0,1,2,3]
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
    U = DM.tensor([random_unitary.random_energy_preserving_unitary(N//num_chunks), DM.Identity(DM.energy_basis(4))])
    system = sim.step(system,ordering[iteration_index],U)




    sub_system = system.ptrace(environment_qbits)
    environment = system.ptrace(sub_system_qbits)
    S1.append(measure.entropy(sub_system))
    S2.append(measure.entropy(environment))
    Stot.append(measure.entropy(system))
    mi = measure.mutual_information(system, sub_system_qbits)
    MI.append(mi)
    if mi <.1:
        print('doing it')
        system.data.toarray().astype(np.complex128).tofile("myTotal.dat")
        sub_system.data.toarray().astype(np.complex128).tofile("mySystem.dat")
        environment.data.toarray().astype(np.complex128).tofile("myEnvironment.dat")


plt.plot(S1, label = "system")
plt.plot(S2, label = "environment")
plt.plot(Stot, label = "Total")
plt.plot(MI,label = "Mutual Information")
plt.legend()
plt.show()
