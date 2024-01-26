import src.density_matrix as DM
import matplotlib.pyplot as plt
import numpy as np
from Scripts import simulation_CLI

data = simulation_CLI.execute(
                connectivity="c5",
                ordering_seed=1,
                unitary_energy_subspace=2,
                unitary_seed=1,
                chunk_size=4,
                num_steps=500,
                initial_pops=[.4, .2, .2, .2, .2, .2, .2, .2],
                evolution_generation_type="hamiltonian"
            )



data  = data[1]
plt.plot(data)
plt.show()

# Extract indices and values from the dataset
# indices = list(data[0].keys())  # Assuming all dictionaries have the same keys
# values_over_time = [[point[index] for index in indices] for point in data]
#
# # Plotting
# for i, index in enumerate(indices):
#     if 0 in index:
#         # Different color for pairs that include 0
#         plt.plot(range(len(data)), [values[i] for values in values_over_time], label=f'Index {index}', color='orange')
#     else:
#         plt.plot(range(len(data)), [values[i] for values in values_over_time], label=f'Index {index}')
#
# # Adding labels and legend
# plt.xlabel('Time')
# plt.ylabel('Values')
# # plt.legend()
#
# # Show the plot
# plt.show()