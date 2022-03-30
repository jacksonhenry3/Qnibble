from Python.random_hamiltonian import random_unitary
import Python.density_matrix as DM
from Python.ket import Basis
import matplotlib.pyplot as plt
import numpy as np

N = 8
chunks = 4

assert N // chunks == N / chunks

block_size = N // chunks

pops = [.1 for _ in range(N)]
pops = np.random.random(N)/2

sys = DM.nqbit(pops)

b = DM.energy_basis(block_size)
I = DM.Identity(b)


def dm_swap(dm, order):
    dm._data = DM.permute_sparse_matrix(dm._data, list(order))
    dm._basis = Basis(tuple(np.array(dm._basis)[order]))


for i in range(100):
    print(i)
    Unitarys = []
    if chunks > 1:
        for chunk_index in range(chunks - 1):
            to_tensor = [I for _ in range(chunks)]
            to_tensor[chunk_index] = random_unitary(block_size)
            U = to_tensor[0].tensor(*to_tensor[1:])

            assert DM.conserves_energy(U)
            energy = [b.energy for b in U.basis]
            nums = [b.num for b in U.basis]

            U.change_to_energy_basis()
            assert DM.conserves_energy(U)
            # U.plot()
            Unitarys.append(U)
    else:
        Unitarys = [random_unitary(block_size)]
    for U in Unitarys:
        sys = U * sys * U.H

sys.plot()
plt.show()

"""
after tensor product U seems fine  (conservs energy) after changing to energy basis, something is wrong. The basis itself seems correct, which means the data must be wrong. is something wrong with the way i am permuting rows/cols?

"""
