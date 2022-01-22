import numpy as np
import qutip as q
# from utils import logm


def temperature(dm: q.Qobj) -> float:
    assert dm.dims == [[2], [2]], f"must be a a density matrix of a single qubit, instead it has dimension {dm.dims}"
    p = dm.data[1, 1]
    return 1 / np.log((1 - p) / p)


def pop_from_temp(t: float) -> float:
    return 1 / (np.e ** (1 / t) + 1)


def avg_temp(dm: q.Qobj) -> float:
    return float(np.mean([temperature(q.ptrace(dm, i)) for i in range(len(dm.dims[0]))]))


def temps(dm: q.Qobj) -> list:
    return [temperature(q.ptrace(dm, i)) for i in range(len(dm.dims[0]))]


def distance(dm1: q.Qobj, dm2: q.Qobj) -> float:
    return (dm1 * (logm(dm1) - logm(dm2))).tr()


def extractable_work(dm1i, dm2i, dm1f, dm2f):
    T = avg_temp
    return T(dm2f) * distance(dm1f, dm2f) - T(dm2i) * distance(dm1i, dm2i)
