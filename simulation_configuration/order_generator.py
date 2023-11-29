# Add directory above current directory to path
import sys
import numpy as np
import os

sys.path.insert(0, '..')

from src import orders

# generate random orders and save them in simulation_configuration/ for each number of qbits from 8,12,and 16

for i in range(100):
    num_iterations = 1000
    for num_qbits in [8, 12, 16]:
        gas_orderings = orders.n_random_gas_orders(num_qbits=num_qbits, n=num_iterations)
        c5_orderings = orders.n_random_c5_orders(num_qbits=num_qbits, n=num_iterations)
        c6_orderings = orders.n_random_c6_orders(num_qbits=num_qbits, n=num_iterations)
        c7_orderings = orders.n_random_c7_orders(num_qbits=num_qbits, n=num_iterations)
        messenger_orderings = orders.n_alternating_messenger_orders(num_qbits=num_qbits, n=num_iterations)

        # use numpy to save the orders as npy files so they can be easily read as numpy arrays
        base_path = f"../simulation_configuration/{num_qbits}qbits/orders/"

        np.save(base_path + f"gas_{i}.npy", gas_orderings) if not os.path.exists(base_path + f"gas_{i}.npy") else print(f"path already exists, skipping gas {num_qbits} qbits")
        np.save(base_path + f"c5_{i}.npy", c5_orderings) if not os.path.exists(base_path + f"c5_{i}.npy") else print(f"path already exists, skipping c5 {num_qbits} qbits")
        np.save(base_path + f"c6_{i}.npy", c6_orderings) if not os.path.exists(base_path + f"c6_{i}.npy") else print(f"path already exists, skipping c6 {num_qbits} qbits")
        np.save(base_path + f"c7_{i}.npy", c7_orderings) if not os.path.exists(base_path + f"c7_{i}.npy") else print(f"path already exists, skipping c7 {num_qbits} qbits")
        np.save(base_path + f"messenger_{i}.npy", messenger_orderings) if not os.path.exists(base_path + f"messenger_{i}.npy") else print(f"path already exists, skipping messenger {num_qbits} qbits")
