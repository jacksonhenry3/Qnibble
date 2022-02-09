from scipy.sparse import coo_matrix, block_diag
import numpy as np
from math import comb
import sys
import matplotlib.pyplot as plt


def nqubit(n) -> coo_matrix:
    return block_diag([np.ones([comb(n, i), comb(n, i)], dtype=np.float32) for i in range(n + 1)])


lst = []
rng = range(2, 12)
for i in rng:
    print(i)
    A = nqubit(i)
    lst.append(A.getnnz())
# plt.plot(list(rng), lst)
plt.imshow(nqubit(12).toarray())
plt.show()
