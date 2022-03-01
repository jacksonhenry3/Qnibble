import numpy as np
# import matplotlib.pyplot as plt
from itertools import permutations, combinations
# import scipy as sp
from Python.density_matrix import DensityMatrix as DM
from Python.ket import energy_basis, canonical_basis, Ket


def accessible_states(state, partition_size):
    return permutations(state)


def partition(lst, n, offset=0):
    lst = lst.data
    L = len(lst)
    lst = np.roll(lst, offset)
    return [lst[i:i + n] for i in range(0, L, n)]


def all_permutations(partition):
    partial_permutations = [permutations(part) for part in partition]

    perms = list(partial_permutations[0])

    for partial_permutation in partial_permutations[1:]:
        perms = ordered_combinations(perms, list(partial_permutation))
    return set(perms)


def ordered_combinations(l1, l2):
    return [e1 + e2 for e1 in l1 for e2 in l2]


def do(num_qbits, partition_size):
    assert num_qbits / partition_size == num_qbits // partition_size

    states = energy_basis(num_qbits)

    possible_to_entangle_with = {}
    mat = np.zeros((2 ** num_qbits, 2 ** num_qbits))
    for state in states:
        print(state)
        for offset in range(partition_size):
            p = partition(state, partition_size, offset=offset)
            p = [Ket(np.roll(perm, -offset)).num for perm in all_permutations(p)]
            possible_to_entangle_with[state.num] = p
            for v in p:
                mat[state.num, v] = 1
    return mat


dm = DM(do(10,5), canonical_basis(10))
# dm.change_to_energy_basis()
# dm.plot()


[[3, 0.66614],
 [4, 0.37453],
 [5, 0.26548],
 [6, 0.21067],
 [7, 0.17099],
 [8, 0.14657],
 [9, 0.12793],
 [10, 0.11294],
 [11, 0.0997],
 [12, 0.09072],
 [13, 0.08446],
 [14, 0.07807],
 [15, 0.07304],
 [16, 0.06636],
 [17, 0.06314],
 [18, 0.06033],
 [19, 0.05564],
 [20, 0.05356],
 [21, 0.04955],
 [22, 0.04836],
 [23, 0.04526],
 [24, 0.04205],
 [25, 0.04202],
 [26, 0.04],
 [27, 0.03927],
 [28, 0.03668],
 [29, 0.03569],
 [30, 0.03457],
 [31, 0.03309],
 [32, 0.03281],
 [33, 0.03092],
 [34, 0.03045],
 [35, 0.02896],
 [36, 0.02899],
 [37, 0.02857],
 [38, 0.02762],
 [39, 0.02682],
 [40, 0.02603],
 [41, 0.02498],
 [42, 0.02455],
 [43, 0.02448],
 [44, 0.02306],
 [45, 0.02296],
 [46, 0.022],
 [47, 0.02187],
 [48, 0.02116],
 [49, 0.02069],
 [50, 0.02179],
 [51, 0.01987],
 [52, 0.01934],
 [53, 0.01948],
 [54, 0.01864],
 [55, 0.01854],
 [56, 0.01862],
 [57, 0.01839],
 [58, 0.01873],
 [59, 0.01687],
 [60, 0.01734],
 [61, 0.01716],
 [62, 0.01677],
 [63, 0.01637],
 [64, 0.01595],
 [65, 0.01572],
 [66, 0.01582],
 [67, 0.01456],
 [68, 0.01495],
 [69, 0.01546],
 [70, 0.01428],
 [71, 0.0146],
 [72, 0.01449],
 [73, 0.01326],
 [74, 0.01448],
 [75, 0.01343],
 [76, 0.01288],
 [77, 0.0127],
 [78, 0.01341],
 [79, 0.01281],
 [80, 0.01269],
 [81, 0.01263],
 [82, 0.01193],
 [83, 0.01245],
 [84, 0.01168],
 [85, 0.01222],
 [86, 0.01197],
 [87, 0.01127],
 [88, 0.01159],
 [89, 0.01165],
 [90, 0.01138],
 [91, 0.01127],
 [92, 0.01121],
 [93, 0.01074],
 [94, 0.01067],
 [95, 0.01094],
 [96, 0.01046],
 [97, 0.00995],
 [98, 0.01056],
 [99, 0.01063]]