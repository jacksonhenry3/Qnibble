import sys as SYS
import matplotlib.pyplot as plt
import numpy as np

SYS.path.insert(0, '..')
SYS.path.insert(0, '')

from src import setup

# setup.use_gpu()
from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    random_unitary,
    orders)

n = 16
qbit = 8

# create a numpy array of x and y indices
x, y = np.indices((2 ** n, 2 ** n))
mask0 = np.bitwise_and((2 ** n - 1) - x, 2 ** (n - qbit - 1)) * np.bitwise_and((2 ** n - 1) - y, 2 ** (n - qbit - 1)) != 0
mask1 = np.bitwise_and(x, 2 ** (n - qbit - 1)) * np.bitwise_and(y, 2 ** (n - qbit - 1)) != 0

data = mask0*2.0
system = DM.DensityMatrix(matrix = data, basis =  DM.canonical_basis(n))





system.change_to_energy_basis()
system.plot()
plt.show()
