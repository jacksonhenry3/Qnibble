from src.random_hamiltonian import random_hamiltonian
from src import density_matrix as DM
from src.ket import energy_basis, canonical_basis
from src import measurements
import matplotlib.pyplot as plt
import numpy as np
from src.step import step
from src.random_hamiltonian import random_unitary

# Properties of the system
number_of_qbits = 16

# initial conditions
initial_pops = np.random.random(number_of_qbits)/2

# generate the system and change to the energy basis
sys = DM.n_thermal_qbits(initial_pops)
sys.change_to_energy_basis()

# how the system will be broken up
chunks = 4

assert number_of_qbits // chunks == number_of_qbits / chunks
block_size = number_of_qbits // chunks

U = random_unitary(4)

Unitaries = [U for _ in range(chunks)]

groupss = [[[0, 1, 2, 3], [4, 5, 6, 7],[8,9,10,11],[12,13,14,15]], [[0, 1, 2, 4], [3, 5, 6, 8],[7,9,10,11],[12,13,14,15]]]

rng = np.random.default_rng()
arr = np.arange(number_of_qbits)

pops = [initial_pops]
means = [[np.mean(initial_pops[:4]), np.mean(initial_pops[4:])]]
import time
start = time.time()
for _ in range(25):
    print(_)
    U = random_unitary(4)

    Unitaries = [U for _ in range(chunks)]
    random_index = np.random.randint(len(groupss))
    groups = groupss[random_index]
    sys = step(sys, groups, Unitaries)
    #pops.append(measurements.pops(sys))
    #means.append([np.mean(pops[-1][:4]), np.mean(pops[-1][4:])])
print(time.time()-start)
#for qbit_index, pop in enumerate(np.transpose(pops)):
#    color = 'b'
#    if qbit_index > 3:
#        color = 'r'
#    plt.plot(pop, c=color, alpha=.2)
#means = np.transpose(means)
#plt.plot(means[0], 'b')
#plt.plot(means[1], 'r')
#plt.legend([f"qbit {i}" for i in range(number_of_qbits)])
#plt.show()
