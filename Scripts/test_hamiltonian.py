import numpy as np
from src import (density_matrix as DM, random_unitary as RU)

random_unitary = RU.random_hamiltonian_in_subspace(4,2)
print(random_unitary.data.toarray())