from src.random_hamiltonian import random_unitary
from src import density_matrix as DM

N = 8
pops = [.1 for _ in range(N)]
pops[2] = .3
sys = DM.nqbit(pops)
b = DM.energy_basis(N//2)
I = DM.Identity(b)

for i in range(100):
    print(i)
    U1 = random_unitary(N//2).tensor(I)
    U1.change_to_energy_basis()
    U2 = I.tensor(random_unitary(N//2))
    U2.change_to_energy_basis()
    sys = U1 * sys * U1.H
    sys = U2 * sys * U2.H
# sys.plot()
# plt.show()
