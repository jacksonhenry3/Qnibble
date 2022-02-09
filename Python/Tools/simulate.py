import numpy as np
import qutip as q
from utils import n_thermal_qubits
from randomHamiltonian import random_hamiltonian
from ThermalProperties import temps


def simulate(nqubits: int, block_size: int, T: int, dm: q.Qobj = None, track_temp: bool = False, tidy: bool = True):
    num_blocks = nqubits // block_size
    assert num_blocks == nqubits / block_size, f"block size {block_size} must evenly divide the number of qubits {nqubits}."
    if not dm:
        dm = n_thermal_qubits(np.random.rand(nqubits))
    order = list(range(nqubits))

    if track_temp: temp_hist = [temps(dm)]

    hi = [random_hamiltonian(block_size, block_size // 2) for i in range(num_blocks)]
    # see https://physics.stackexchange.com/questions/164109/dealing-with-tensor-products-in-an-exponent
    U = q.tensor(*[(-1j * .1 * h).expm() for h in hi])

    for t in range(T):
        O = list(np.roll(order, np.random.randint(0, nqubits)))
        U = U.permute(O)
        dm = (U * dm * U.dag())

        if track_temp: temp_hist.append(temps(dm))
        if tidy: dm.tidyup()

    if track_temp: return dm, temp_hist
    return dm

# print('done')
# plt.imshow(np.ceil(np.real(dm)), cmap='gray', interpolation='none')
# # plt.imshow(np.real(dm), cmap='PuOr', interpolation='none')
# plt.axis('off')
# plt.tight_layout()
# plt.show()
