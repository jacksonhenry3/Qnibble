import sys
import numpy as np
import os
import pickle

sys.path.insert(0, '..')

from src import random_unitary, density_matrix as DM

identity = DM.Identity(DM.energy_basis(4))
sub_unitary = random_unitary.random_energy_preserving_unitary(4)

# generate random unitaries and save them in simulation_configuration/ for each number of qbits from 8,12,and 16
for num_qbits in [8, 12, 16]:

    # use numpy to save the unitaries as npy files so they can be easily read as numpy arrays
    path = f"../simulation_configuration/{num_qbits}qbits/unitaries/same_sub_unitary_fully_random.pkl"

    # check if the path already exists, if it does, skip it
    if os.path.exists(path):
        print(f"path already exists, skipping unitary {num_qbits} qbits")
        continue

    else:
        num_chunks = num_qbits // 4

        composite_unitaries = [DM.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in range(num_chunks)]
        unitary = np.product(composite_unitaries)
        with open(path, 'wb') as file:
            pickle.dump(unitary, file)
