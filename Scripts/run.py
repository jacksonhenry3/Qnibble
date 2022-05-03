from src.random_hamiltonian import random_hamiltonian
from src import density_matrix as DM
from src.ket import energy_basis, canonical_basis
from src import measurements
import matplotlib.pyplot as plt
import numpy as np
from src.step import step
from src.random_hamiltonian import random_unitary

# Properties of the system
number_of_qbits = 8

# initial conditions
initial_pops = [.1, .1, .1, .1, .1, .1, .2, .1]

# generate the system and change to the energy basis
sys = DM.n_thermal_qbits(initial_pops)
sys.change_to_energy_basis()

# how the system will be broken up
chunks = 2

assert number_of_qbits // chunks == number_of_qbits / chunks
block_size = number_of_qbits // chunks

U = random_unitary(4)

Unitaries = [U for _ in range(chunks)]

groupss = [[[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2, 4], [3, 5, 6, 7]]]
#groupss = [[[0, 1, 4, 2], [3, 5, 6, 7]], [[0, 1, 4,2], [3, 5, 6, 7]]]
#groupss = [[[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 4, 2], [3, 5, 6, 7]]]

pops = [initial_pops]
for _ in range(100):
    print(_)
    random_index = np.random.randint(2)
    groups = groupss[random_index]
    sys = step(sys, groups, Unitaries)
    pops.append(measurements.pops(sys))


for qbit_index,pop in enumerate(np.transpose(pops)):
    # color = 'b'
    # if qbit_index>3:
    #     color = 'r'
    plt.plot(pop)
plt.legend([f"qbit {i}" for i in range(number_of_qbits)])
plt.show()
