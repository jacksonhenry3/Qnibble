import numpy as np
from density_matrix import DensityMatrix, nqbit, dm_trace, dm_log


# measurements
def temp(qbit: DensityMatrix):
    assert qbit.size == 2, "density matrix must be for a single qubit"
    p = qbit.data[1, 1]
    return temp_from_pop(p)


def temps(dm: DensityMatrix):
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        to_trace = list(range(n))
        to_trace.remove(i)
        result.append(temp(dm.ptrace(to_trace)))
    return result


def temp_from_pop(pop: float):
    return 1 / (np.log((1 - pop) / pop))


def pop_from_temp(T: float):
    return 1 / (1 + np.exp(1 / T))


def D(dm1: DensityMatrix, dm2: DensityMatrix):
    assert dm1.size == dm2.size
    return dm_trace(dm1 * dm_log(dm1)) - dm_trace(dm1 * dm_log(dm2))


def extractable_work(T: float, dm: DensityMatrix):
    pop = pop_from_temp(T)
    reference_dm = nqbit([pop for _ in range(dm.number_of_qbits)])
    return T * D(dm, reference_dm)


def change_in_extractable_work(T_initial: float, dm_initial: DensityMatrix, T_final: float, dm_final: DensityMatrix):
    return extractable_work(T_final, dm_final) - extractable_work(T_initial, dm_initial)


def entropy(dm: DensityMatrix):
    return dm_trace(-dm * dm_log(dm))
