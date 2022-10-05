from src import density_matrix as DM
from src import measurements
import matplotlib.pyplot as plt
import numpy as np
from src.step import step
import cupy as cp
from src.random_unitary import random_unitary
import src.measurements as measure

print("all modules loaded")
rng = np.random.default_rng(seed=0)

# mempool = cp.get_default_memory_pool()
# mempool.set_limit(size=16 * 1024 ** 3)
# # Properties of the system
# number_of_qbits = 16
#
#
# # initial conditions
# initial_pops = np.random.random(number_of_qbits)
# num_blocks = 8
# # generate the system and change to the energy basis
# sys = DM.n_thermal_qbits(initial_pops)
# sys.change_to_energy_basis()


# block_size = number_of_qbits // num_blocks

a = DM.n_thermal_qbits([.3, .2, .3, .4])
print(a)
# import time
#
# start = time.time()
# for i in range(100):
#     print(f"using {mempool.used_bytes()} out of a set limit of {mempool.total_bytes()}, {mempool.get_limit()}")
#     print(i)
#     print(sys.data.nnz)
#     sub_system_unitaries = [random_unitary(block_size) for _ in range(num_blocks)]
#
#     U = DM.tensor(sub_system_unitaries)
#     sub_system_unitaries = None
#     # shift the order of the qbits
#
#     order = rng.permutation(number_of_qbits)
#
#     U.relabel_basis(order)
#     U.change_to_energy_basis()
#
#     mempool.free_all_blocks()
#     sys = U * sys
#     mempool.free_all_blocks()
#     U = U.H
#     mempool.free_all_blocks()
#     sys = sys * U
#     mempool.free_all_blocks()
# print(time.time() - start)
#
# measurments = [np.vstack((measurments[i], measurment(qm_sys))) for i, measurment in enumerate(measurment_set)]
# print(f"{np.round(time.time() - start, 2)} seconds elapsed")
# return measurments

# cpu 10 runs 13.5690336227417
# gpu 10 runs 4.858150005340576
