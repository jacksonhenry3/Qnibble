from Python.random_hamiltonian import random_unitary
import Python.density_matrix as DM
import matplotlib.pyplot as plt

N = 10
pops = [.1 for _ in range(N)]
pops[2] = .3
sys = DM.nqbit(pops)
for i in range(100):
    print(i)
    U = random_unitary(N)
    sys = U * sys * U.H
# sys.plot()
# plt.show()
