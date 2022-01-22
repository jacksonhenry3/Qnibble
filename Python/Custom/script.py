import DensityMatrix as DM
import numpy as np
from RandomHamiltonian import random_hamiltonian
import matplotlib.pyplot as plt
from Ket import canonical_basis, energy_basis

N = 6
system = DM.nqbit([np.random.random() for i in range(N)])

for i in range(100):
    print(i)
    H = random_hamiltonian(3).tensor(DM.Identity(3)) + DM.Identity(3).tensor(random_hamiltonian(3))
    # assert U.basis == energy_basis(6)
    # H.change_to_energy_basis()
    assert H.basis == energy_basis(6)
    U = DM.exp(H * .1j)
    U.change_to_energy_basis()
    # system.change_to_energy_basis()


    system = U * system
    system = system * U.H
system.change_to_canonical_basis()
system.plot()
