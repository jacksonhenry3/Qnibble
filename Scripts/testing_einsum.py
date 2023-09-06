import numpy as np
from src import (density_matrix as DM)
import itertools

dm1 = DM.DensityMatrix(np.random.random((4, 4)), DM.canonical_basis(2))
dm2 = DM.DensityMatrix(np.random.random((4, 4)), DM.canonical_basis(2))

product = dm1.tensor(dm2)

base_str = "abIJefIJ"
for order in itertools.permutations(base_str, 8):
    order = "".join(order)
    dm1_extracted = np.einsum(order, product.data.toarray().reshape((2, 2, 2, 2, 2, 2, 2, 2))).reshape(4, 4)
    if np.allclose(dm1.data.toarray(), dm1_extracted):
        print(order, "dm1")
    if np.allclose(dm2.data.toarray(), dm1_extracted):
        print(order, "dm2")
