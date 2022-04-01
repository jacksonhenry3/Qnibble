import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


from Python.random_hamiltonian import random_unitary
import Python.density_matrix as DM
from Python.ket import Basis
import matplotlib.pyplot as plt
import numpy as np
import Python.measurements as measure

N = 15
chunks = 3

assert N // chunks == N / chunks

block_size = N // chunks

pops = [.1 for _ in range(N)]
pops[3] = .4

sys = DM.n_thermal_qbits(pops)
sys.change_to_energy_basis()
b = DM.energy_basis(block_size)
I = DM.Identity(b)


def dm_swap(dm, order):
    dm._data = DM.permute_sparse_matrix(dm._data, list(order))
    dm._basis = Basis(tuple(np.array(dm._basis)[order]))

temps = []
for i in range(100):
    print(i)
    Unitarys = []
    if chunks > 1:
        for chunk_index in range(chunks - 1):
            to_tensor = [I for _ in range(chunks)]
            to_tensor[chunk_index] = random_unitary(block_size, dt = .01)
            U = to_tensor[0].tensor(*to_tensor[1:])
            U.change_to_energy_basis()
            Unitarys.append(U)
    else:
        Unitarys = [random_unitary(block_size)]
    for U in Unitarys:
        sys = U * sys * U.H
    temps.append(np.real(measure.temps(sys)))


img = plt.imshow(np.transpose(temps), interpolation="nearest", aspect = 'auto')
img.set_cmap('hot')
plt.axis('off')
plt.show()
