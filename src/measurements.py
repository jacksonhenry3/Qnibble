import scipy.sparse.linalg

import src.setup as setup

xp = setup.xp
sp = setup.sp
SPARSE_TYPE = setup.SPARSE_TYPE

import numpy as np
import src.density_matrix as DM
from src.density_matrix import DensityMatrix, n_thermal_qbits, dm_trace, dm_log, qbit

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
    pop = 1 / (1 + np.exp(1 / T))
    # assert 0 <= pop <= 1, "pop must be between 0 and 1"
    return pop


# /relative entropy
def D(dm1: DensityMatrix, dm2: DensityMatrix):
    assert dm1.size == dm2.size

    # this is related to entropy
    return dm_trace(dm1 * dm_log(dm1)) - dm_trace(dm1 * dm_log(dm2))


def D_single_qbits(pop_1: float, pop_2: float):
    tr_1 = (1 - pop_1) * np.log(1 - pop_1) + (pop_1) * np.log(pop_1)
    tr_2 = (1 - pop_2) * np.log(1 - pop_2) + (pop_2) * np.log(pop_2)
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


# von neumann entropy
def entropy(dm: DensityMatrix) -> float:
    # if dm.number_of_qbits <= 2:
    # result = -dm_trace(dm * dm_log(dm))
    # return float(xp.real(result))


    if dm.number_of_qbits == 1:
        eigen_vals = dm.data.diagonal()
        from_eigen = -np.sum(eigen_vals * np.log(eigen_vals))
    elif dm.number_of_qbits == 2:
        a1 = dm.data[0, 0]
        b2 = dm.data[1, 1]
        b3 = dm.data[1, 2]
        c2 = dm.data[2, 1]
        c3 = dm.data[2, 2]
        d4 = (1 - a1 - b2 - c3)

        from_eigen = (
                a1 * np.log(a1) +
                d4 * np.log(d4) +
                0.5 * (b2 + c3 - np.sqrt(b2 ** 2 + 4 * b3 * c2 - 2 * b2 * c3 + c3 ** 2)) * np.log(0.5 * (b2 + c3 - np.sqrt(b2 ** 2 + 4 * b3 * c2 - 2 * b2 * c3 + c3 ** 2))) +
                0.5 * (b2 + c3 + np.sqrt(b2 ** 2 + 4 * b3 * c2 - 2 * b2 * c3 + c3 ** 2)) * np.log(0.5 * (b2 + c3 + np.sqrt(b2 ** 2 + 4 * b3 * c2 - 2 * b2 * c3 + c3 ** 2)))
        )

    else:
        # This method can't find all eigenvalues becouse of the algorithm it uses, but it does find all but the smallest two,
        # leading to a precision loss of ~10-6
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html for details
        if setup.using_gpu:
            eigen_vals = scipy.sparse.linalg.eigsh(dm.data.get(), k=2 ** dm.number_of_qbits - 3, which="LM", return_eigenvectors=False)
            from_eigen = -np.sum(eigen_vals * np.log(eigen_vals))
        else:
            eigen_vals = sp.sparse.linalg.eigsh(dm.data, k=2 ** dm.number_of_qbits - 3, return_eigenvectors=False)
            from_eigen = -np.sum(eigen_vals * np.log(eigen_vals))
    return np.real(from_eigen)


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
    vals, _ = sp.sparse.linalg.eig(data @ spin_flipped)

    sorted_sqrt_eig_vals = np.real(np.sort(np.sqrt(vals)))
    combined = sorted_sqrt_eig_vals[3] - sorted_sqrt_eig_vals[2] - sorted_sqrt_eig_vals[1] - sorted_sqrt_eig_vals[0]
    return max(combined, 0)


def uncorrelated_thermal_concurrence(dm: DensityMatrix) -> float:
    a = dm.data.toarray()[1, 2]
    b = dm.data.toarray()[0, 0]
    c = dm.data.toarray()[3, 3]
    return np.abs(a) - np.sqrt(b * c)


def mutual_information_with_environment(dm: DensityMatrix, sub_system_qbits: list[int]) -> float:
    environment_qbits = list(set(range(dm.basis.num_qubits)) - set(sub_system_qbits))
    sub_system = dm.ptrace(environment_qbits)
    environment = dm.ptrace(sub_system_qbits)
    return entropy(sub_system) + entropy(environment) - entropy(dm)


def mutual_information(dm: DensityMatrix, sub_system_qbits_a: list[int], sub_system_qbits_b: list[int]) -> float:
    everything_thats_not_system_a = tuple(set(range(dm.basis.num_qubits)) - set(sub_system_qbits_a))
    sub_system_a = dm.ptrace(everything_thats_not_system_a)

    everything_thats_not_system_b = tuple(set(range(dm.basis.num_qubits)) - set(sub_system_qbits_b))
    sub_system_b = dm.ptrace(everything_thats_not_system_b)

    sub_system_qbits_ab = sub_system_qbits_a + sub_system_qbits_b
    everything_thats_not_system_ab = tuple(set(range(dm.basis.num_qubits)) - set(sub_system_qbits_ab))
    sub_system_ab = dm.ptrace(everything_thats_not_system_ab)
    return entropy(sub_system_a) + entropy(sub_system_b) - entropy(sub_system_ab)


def mutual_information_of_every_pair(dm: DensityMatrix):
    n = dm.number_of_qbits
    result = {}
    for i in range(n):
        for j in range(i + 1, n):
            result[(i,j)] = mutual_information(dm, [i], [j])
    return np.array(result)


def relative_entropy_of_every_pair(dm: DensityMatrix):
    n = dm.number_of_qbits
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            dm1 = DM.qbit(dm.ptrace_to_a_single_qbit(i))
            dm2 = DM.qbit(dm.ptrace_to_a_single_qbit(j))
            result.append((D(dm1, dm2), i, j))
    return result


# monogamy of Mutual Information
def monogamy_of_mutual_information(dm: DensityMatrix, sub_system_qbits_a: list[int], sub_system_qbits_b: list[int], sub_system_qbits_c: list[int]) -> float:
    dm_abc = dm
    dm_ab = dm.ptrace(sub_system_qbits_c)
    dm_ac = dm.ptrace(sub_system_qbits_b)
    dm_bc = dm.ptrace(sub_system_qbits_a)
    dm_a = dm.ptrace(sub_system_qbits_b + sub_system_qbits_c)
    dm_b = dm.ptrace(sub_system_qbits_a + sub_system_qbits_c)
    dm_c = dm.ptrace(sub_system_qbits_a + sub_system_qbits_b)
    s = entropy
    return s(dm_abc) + s(dm_a) + s(dm_b) + s(dm_c) - s(dm_ab) - s(dm_ac) - s(dm_bc)


# subaddativity
def subaddativity(dm: DensityMatrix, sub_system_qbits_a: list[int], sub_system_qbits_b: list[int]) -> float:
    dm_ab = dm
    dm_a = dm.ptrace(sub_system_qbits_b)
    dm_b = dm.ptrace(sub_system_qbits_a)
    s = entropy
    return -s(dm_ab) + s(dm_a) + s(dm_b)


def strong_subaddativity(dm: DensityMatrix, sub_system_qbits_a: list[int], sub_system_qbits_b: list[int], sub_system_qbits_c: list[int]) -> float:
    dm_ab = dm.ptrace(sub_system_qbits_c)
    dm_bc = dm.ptrace(sub_system_qbits_a)
    dm_a = dm.ptrace(sub_system_qbits_b + sub_system_qbits_c)
    dm_c = dm.ptrace(sub_system_qbits_a + sub_system_qbits_b)
    s = entropy
    return s(dm_ab) + s(dm_bc) - s(dm_a) - s(dm_c)
