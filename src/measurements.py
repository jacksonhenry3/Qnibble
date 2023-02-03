import numpy as np
from src.density_matrix import DensityMatrix, n_thermal_qbits, dm_trace, dm_log,qbit
# import cupy as cp
import scipy as sp
import scipy.linalg

σx = np.matrix([[0, 1], [1, 0]])
σy = np.matrix([[0, -1j], [1j, 0]])
σz = np.matrix([[1, 0], [0, -1]])


# measurements
def temp(qbit: DensityMatrix):
    assert qbit.size == 2, "density matrix must be for a single qubit"
    p = pop(qbit)
    return np.real(temp_from_pop(p))


def pop(qbit: DensityMatrix):
    assert qbit.size == 2, "density matrix must be for a single qubit"
    p = qbit.data.diagonal()[1]
    return float(np.real(p))


def pops(dm: DensityMatrix):
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        p = dm.ptrace_to_a_single_qbit(i)
        result.append(p)
    return result


def temps(dm: DensityMatrix):
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        result.append(float(temp_from_pop(dm.ptrace_to_a_single_qbit(i))))
    return result


def average_temp(dm: DensityMatrix) -> float:
    return float(np.mean(temps(dm)))


def temp_from_pop(pop: float):
    return 1 / (np.log((1 - pop) / pop))


def pop_from_temp(T: float):
    return 1 / (1 + np.exp(1 / T))


def D(dm1: DensityMatrix, dm2: DensityMatrix):
    assert dm1.size == dm2.size
    return dm_trace(dm1 * dm_log(dm1)) - dm_trace(dm1 * dm_log(dm2))

def D_single_qbits(pop_1:float,pop_2:float):
    tr_1 = (1-pop_1)*np.log(1-pop_1)+(pop_1)*np.log(pop_1)
    tr_2 = (1-pop_2)*np.log(1-pop_2)+(pop_2)*np.log(pop_2)
    return tr_1 - tr_2


def extractable_work(T: float, dm: DensityMatrix):
    pop = pop_from_temp(T)
    reference_dm = n_thermal_qbits([pop for _ in range(dm.number_of_qbits)])
    reference_dm.change_to_energy_basis()
    dm.change_to_energy_basis()
    return float(np.real(T * D(dm, reference_dm)))

def extractable_work_of_a_single_qbit(T: float, pop: float):
    ref_pop = pop_from_temp(T)
    return float(np.real(T * D_single_qbits(pop, ref_pop)))


def extractable_work_of_each_qubit(dm: DensityMatrix):
    # TODO this breaks when initial state is non-thermal
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        temp_list = temps(dm)
        temp_list.pop(i)
        T = np.mean(temp_list)
        result.append(extractable_work_of_a_single_qbit(T, dm.ptrace_to_a_single_qbit(i)))
    return result


def change_in_extractable_work(T_initial: float, dm_initial: DensityMatrix, T_final: float, dm_final: DensityMatrix):
    return extractable_work(T_final, dm_final) - extractable_work(T_initial, dm_initial)


def entropy(dm: DensityMatrix):
    return dm_trace(-dm * dm_log(dm))


def concurrence(dm: DensityMatrix) -> float:
    """

    Args:
        dm: a 2 qbit density matrix

    Returns: a real number between zero and 1


    ref: https://www.rintonpress.com/journals/qic-1-1/eof2.pdf pg 33

    """

    # TODO assert dm size

    data: np.ndarray = dm.data.toarray()

    spin_flip_operator = sp.linalg.kron(σy, σy)
    spin_flipped = spin_flip_operator @ data.conj() @ spin_flip_operator
    vals, _ = sp.linalg.eig(data @ spin_flipped)

    sorted_sqrt_eig_vals = np.real(np.sort(np.sqrt(vals)))
    combined = sorted_sqrt_eig_vals[3] - sorted_sqrt_eig_vals[2] - sorted_sqrt_eig_vals[1] - sorted_sqrt_eig_vals[0]
    return max(combined, 0)


def uncorrelated_thermal_concurrence(dm: DensityMatrix) -> float:
    a = dm.data.toarray()[1, 2]
    b = dm.data.toarray()[0, 0]
    c = dm.data.toarray()[3, 3]
    return np.abs(a) - np.sqrt(b * c)
