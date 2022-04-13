from src.random_hamiltonian import random_unitary
import matplotlib.pyplot as plt
import numpy as np
from src import measurements as measure, density_matrix as DM

N = 8
chunks = 2

assert N // chunks == N / chunks

block_size = N // chunks

pops = [.1 for _ in range(N)]
pops[3] = .4

sys = DM.n_thermal_qbits(pops)
sys.change_to_energy_basis()

b = DM.energy_basis(block_size)
I = DM.Identity(b)

new_basis = b.tensor(*[b for _ in range(chunks - 1)])

temps = []
sys.change_to_energy_basis()
for i in range(500):
    print(i)
    Unitarys = []

    if chunks > 1:
        for chunk_index in range(chunks - 1):
            to_tensor = [I for _ in range(chunks)]
            to_tensor[chunk_index] = random_unitary(block_size, dt=.01)
            U = to_tensor[0].tensor(*to_tensor[1:])
            order = list(range(N))
            shift = np.random.randint(len(order))
            order = np.roll(order, shift)
            # np.random.shuffle(order)
            U.relabel_basis(order)
            U.change_to_energy_basis()
            Unitarys.append(U)

    else:
        U = random_unitary(block_size)
        U.change_to_energy_basis()
        Unitarys = [U]
    for U in Unitarys:
        # U.change_to_energy_basis()
        # sys.change_to_energy_basis()

        sys = U * sys
        sys = sys * U.H


    temps.append(np.real(measure.temps(sys)))

img = plt.imshow(np.transpose(temps), interpolation="nearest", aspect='auto')
img.set_cmap('hot')
plt.axis('off')
# sys.plot()
plt.show()
