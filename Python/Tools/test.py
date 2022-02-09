import qutip as q
from randomHamiltonian import random_hamiltonian, full_hamiltonian
from utils import n_thermal_qubits, I
import numpy as np
import matplotlib.pyplot as plt
import simulate
# dm = simulate.simulate(nqubits=4, block_size=4, T=50)
dm = random_hamiltonian(nqubit=4,energy=1)+random_hamiltonian(nqubit=4,energy=2)+random_hamiltonian(nqubit=4,energy=3)
# plt.imshow(np.ceil(np.real(dm)), cmap='gray', interpolation='none')
# plt.imshow(np.real(dm), cmap='PuOr', interpolation='none')
q.hinton(dm)
# plt.axis('off')
# plt.tight_layou/t()
plt.show()