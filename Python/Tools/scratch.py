import qutip as q
from randomHamiltonian import random_hamiltonian, full_hamiltonian
from utils import n_thermal_qubits, I
import matplotlib.pyplot as plt

# obj = full_hamiltonian(nqubit=3, energy=1) + full_hamiltonian(nqubit=3, energy=2)

obj = q.Qobj([[1, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 0, 0, 0, 0], [0, 2, 2, 2, 0, 0, 0, 0], [0, 2, 2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 2, 0], [0, 0, 0, 0, 2, 2, 2, 0], [0, 0, 0, 0, 2, 2, 2, 0],
              [0, 0, 0, 0, 0, 0, 0, 3]], dims=[[2,2,2],[4,2]])
q.hinton(obj)
plt.show()

# 1 + 2 + 1 = 2 * 2
# 1 + 3 + 3 + 1 = 2 * 2 * 2
# 1 + 4 + 6 + 4 + 1 = 2 * 2 * 2 * 2
# new = q.Qobj()
# q.hinton(q.tensor([q.thermal_dm(1, .1, "analytic"), q.thermal_dm(2, .1, "analytic"), q.thermal_dm(1, .1, "analytic")]))
# plt.show()
