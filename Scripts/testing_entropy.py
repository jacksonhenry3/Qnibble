import numpy as np

import random_unitary
from src import (density_matrix as DM)

dm1 = DM.n_thermal_qbits([.1, .2])
U = random_unitary.random_energy_preserving_unitary(2)
dm1 = U * dm1 * U.H
dm2 = DM.n_thermal_qbits([.3, .4])
U = random_unitary.random_energy_preserving_unitary(2)
dm2 = U * dm2 * U.H

product = dm1.tensor(dm2)

# print(np.allclose(dm2.data.toarray(), product.ptrace([0, 0]).data.toarray()))
print(np.allclose(dm2.data.toarray(), product.ptrace([0, 1]).data.toarray()))

print("===")

print(np.allclose(dm1.data.toarray(), product.ptrace([2, 3]).data.toarray()))