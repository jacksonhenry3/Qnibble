from Python.random_hamiltonian import random_unitary, random_hamiltonian
import Python.density_matrix as DM
import matplotlib.pyplot as plt
import numpy as np

pops = [.1, .2]
sys = DM.nqbit(pops)
data = sys.data.toarray()
print(data @ data)
test = sys.qbit_basis()

print(test)
