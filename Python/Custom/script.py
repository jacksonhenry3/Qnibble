import DensityMatrix as DM
import numpy as np
from RandomHamiltonian import random_hamiltonian
import matplotlib.pyplot as plt

# N = 8
# system = DM.nqbit([np.random.random() for i in range(N)])
#
# #
# #
# # for i in range(100):
# #     print(i)
# #     H = random_hamiltonian(N)
# #     U = DM.exp(H * .1j)
# #     system = (U * system * U.H)
# # # system.change_to_canonical_basis()
# # system.plot()
# H = random_hamiltonian(4).tensor(DM.Identity(4))+DM.Identity(4).tensor(random_hamiltonian(4))
# # H.change_to_energy_basis()
# H.plot()
a = np.random.random((2, 2))
b = np.random.random((2, 2))
b = np.identity(2)
# a = np.identity(2)
print(a.ravel())
c = np.einsum('i,j->ij', a.ravel(), b.ravel())
d = np.outer(a, b)
print(c == d)
plt.show()
