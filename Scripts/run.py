from src.random_hamiltonian import random_unitary
from src import density_matrix as DM
import numpy as np
N = 4
pops = [.1 for _ in range(N)]
pops[2] = .3
sys1 = DM.n_thermal_qbits(pops)
sys2 = DM.n_thermal_qbits(pops)
sys1.plot()
sys1.relabel_basis([0, 1, 3, 2])
sys1.change_to_canonical_basis()
sys1.plot()

sys1 * sys2
