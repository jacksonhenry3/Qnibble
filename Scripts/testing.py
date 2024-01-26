import sys

sys.path.insert(0, '..')

import numpy as np

from src import setup
# setup.use_gpu()

from src import (
    measurements as measure,
    density_matrix as DM,
    simulation as sim,
    orders,
    random_unitary)

import simulation_CLI as cleo

data=cleo.execute("gas", 3, 2, 2,4,100, [0.4, 0.2, 0.2, 0.2])

print(data[0])

print(measure.mutual_information_of_every_pair_dict(data[0]))

