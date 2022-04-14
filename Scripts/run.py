from src.random_hamiltonian import random_hamiltonian
from src import density_matrix as DM
from src.ket import energy_basis
from src import measurements
import matplotlib.pyplot as plt
import numpy as np
from src.step import step

dtheta = .025

# Properties of the system
number_of_qbits = 8

# initial conditions
pops = [.2 for _ in range(number_of_qbits)]
pops[5] = .4

# generate the system and change to the energy basis
sys = DM.n_thermal_qbits(pops)
sys.change_to_energy_basis()

# how the system will be broken up
chunks = 2
assert number_of_qbits // chunks == number_of_qbits / chunks
block_size = number_of_qbits // chunks

data = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0. + 0.0585211j, 0., 0. + 0.06734j, 0., 0., 0.,
     0. + 0.5487j, 0., 0., 0., 0., 0., 0., 0.],
    [0., 0. - 0.0585211j, 0., 0., 0. + 0.824991j, 0., 0., 0.,
     0. + 0.524563j, 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0. + 0.0687147j, 0. + 0.372811j, 0., 0.,
     0. + 0.54036j, 0. + 0.884018j, 0., 0. + 0.574957j, 0., 0., 0.],
    [0., 0. - 0.06734j, 0. - 0.824991j, 0., 0., 0., 0., 0.,
     0. + 0.594644j, 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0. - 0.0687147j, 0., 0., 0. + 0.43009j, 0., 0.,
     0. + 0.0999313j, 0. + 0.27748j, 0., 0. + 0.128059j, 0., 0., 0.],
    [0., 0., 0., 0. - 0.372811j, 0., 0. - 0.43009j, 0., 0., 0.,
     0. + 0.938269j, 0. + 0.953751j, 0., 0. + 0.590938j, 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. + 0.0901577j, 0.,
     0. + 0.517302j, 0. + 0.474071j, 0.],
    [0., 0. - 0.5487j, 0. - 0.524563j, 0., 0. - 0.594644j, 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0. - 0.54036j, 0., 0. - 0.0999313j, 0. - 0.938269j,
     0., 0., 0., 0. + 0.623484j, 0., 0. + 0.874511j, 0., 0., 0.],
    [0., 0., 0., 0. - 0.884018j, 0., 0. - 0.27748j, 0. - 0.953751j,
     0., 0., 0. - 0.623484j, 0., 0., 0. + 0.137632j, 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0. - 0.0901577j, 0., 0., 0., 0., 0.,
     0. + 0.948898j, 0. + 0.915149j, 0.],
    [0., 0., 0., 0. - 0.574957j, 0., 0. - 0.128059j, 0. - 0.590938j,
     0., 0., 0. - 0.874511j, 0. - 0.137632j, 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0. - 0.517302j, 0., 0., 0.,
     0. - 0.948898j, 0., 0., 0. + 0.0418653j, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0. - 0.474071j, 0., 0., 0.,
     0. - 0.915149j, 0., 0. - 0.0418653j, 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
]

# H = DM.DensityMatrix(DM.SPARSE_TYPE(data), canonical_basis(block_size))


pops = []

possible_groups = []
for offset in range(4):
    order = list(range(number_of_qbits))
    order = np.roll(order, offset)
    order = order.reshape((2,4))
    possible_groups.append(order)



for _ in range(500):
    print(_)

    H = random_hamiltonian(block_size)
    groups = possible_groups[np.random.randint(4)]
    step_sizes = [.025, .021]

    sys = step(sys, groups, [H, H], step_sizes)

    pops.append(measurements.pops(sys))

# plt.imshow(np.transpose(pops))
plt.plot(pops)
plt.show()

plt.plot(np.sum(pops, 1))
plt.show()
