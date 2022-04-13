import numpy as np
from density_matrix import DensityMatrix, n_thermal_qbits, dm_trace, dm_log


# measurements
def temp(qbit: DensityMatrix):
    assert qbit.size == 2, "density matrix must be for a single qubit"
    p = qbit.data.diagonal()[1]
    return np.real(temp_from_pop(p))

def pop(qbit: DensityMatrix):
    assert qbit.size == 2, "density matrix must be for a single qubit"
    p = qbit.data.diagonal()[1]
    return p

def pops(dm: DensityMatrix):
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        index = dm.basis[0]._order[i]
        result.append(pop(dm.ptrace_to_a_single_qbit(index)))
    return result

def temps(dm: DensityMatrix):
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        index = dm.basis[0]._order[i]
        result.append(temp(dm.ptrace_to_a_single_qbit(index)))
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


def extractable_work(T: float, dm: DensityMatrix):
    pop = pop_from_temp(T)
    reference_dm = n_thermal_qbits([pop for _ in range(dm.number_of_qbits)])
    return T * D(dm, reference_dm)


def extractable_work_of_each_qubit(dm: DensityMatrix):
    # TODO this breaks when initial state is non-thermal
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        index = dm.basis[0]._order[i]
        temp_list = temps(dm)
        temp_list.pop(i)
        T = np.mean(temp_list)
        result.append(extractable_work(T, dm.ptrace_to_a_single_qbit(index)))
    return result


def change_in_extractable_work(T_initial: float, dm_initial: DensityMatrix, T_final: float, dm_final: DensityMatrix):
    return extractable_work(T_final, dm_final) - extractable_work(T_initial, dm_initial)


def entropy(dm: DensityMatrix):
    return dm_trace(-dm * dm_log(dm))
