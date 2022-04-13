from src.random_hamiltonian import random_hamiltonian
from src import density_matrix as DM
from src.ket import energy_basis
from src import measurements
import matplotlib.pyplot as plt
import numpy as np

dtheta = .025

# Properties of the system
number_of_qbits = 8

# initial conditions
pops = [.2 for _ in range(number_of_qbits)]
pops[3] = .4

# generate the system and change to the energy basis
sys = DM.n_thermal_qbits(pops)
sys.change_to_energy_basis()

# how the system will be broken up
chunks = 2
assert number_of_qbits // chunks == number_of_qbits / chunks
block_size = number_of_qbits // chunks

H = random_hamiltonian(block_size)
I = DM.Identity(energy_basis(block_size))
H = H.tensor(I) + I.tensor(H)
U = DM.dm_exp(-H * dtheta * 1j)

pops = []
for _ in range(500):
    print(_)

    order = list(range(number_of_qbits))
    shift = np.random.randint(len(order))
    order = np.roll(order, shift)
    # np.random.shuffle(order)
    U.relabel_basis(order)
    U.change_to_energy_basis()
    assert sys.basis is not U.basis
    sys = U * sys
    sys = sys * U.H
    assert sys.basis is not U.basis
    pops.append(measurements.pops(sys))

plt.plot(pops)
plt.show()
