from src import density_matrix as dm
from src import random_hamiltonian as r
from src import ket
#pops = [.2,.2,]
#sys = dm.n_thermal_qbits(pops)
#sys.change_to_energy_basis()
#sys.plot()
#data = sys.data.toarray()
#print(data @ data)
#test = sys.qbit_basis()

u1 = r.random_unitary(4)
u2 = r.random_unitary(4)
U1 = dm.Identity(ket.canonical_basis(4)).tensor(u1)
U2 = u2.tensor(dm.Identity(ket.canonical_basis(4)))
U1.change_to_energy_basis()
U2.change_to_energy_basis()
U = U1*U2
U.change_to_energy_basis()
U1.plot()
U2.plot()
#print(test)
