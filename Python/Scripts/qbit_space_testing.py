from Python.random_hamiltonian import random_unitary, random_hamiltonian
import Python.density_matrix as DM
import matplotlib.pyplot as plt
import numpy as np

N = 6
chunks = 2

assert N // chunks == N / chunks


block_size = N//chunks

pops = [.1 for _ in range(N)]

sys = DM.nqbit(pops)

b = DM.energy_basis(block_size)
I = DM.Identity(b)


for i in range(100):
    print(i)
    Unitarys = []
    for chunk_index in range(chunks):
        to_tensor = [I for _ in range(chunks)]
        to_tensor[chunk_index] = random_unitary(block_size)
        U = to_tensor[0].tensor(*to_tensor[1:])
        # U.change_to_energy_basis()
        Unitarys.append(U)

    for U in Unitarys:
        sys = U * sys * U.H

sys.plot()
plt.show()
