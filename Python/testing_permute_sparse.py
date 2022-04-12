import numpy as np
import Python.density_matrix as DM
import scipy.sparse as sp
import matplotlib.pyplot as plt
from measurements import temps
for i in range(100):
    print(i)
    pops = [.1, .1, .4]
    therm2 = DM.n_thermal_qbits(pops)
    therm2.change_to_energy_basis()
    temps(therm2)
    # print(therm2.ptrace([0, 1]))
    # print(therm2.ptrace_to_a_single_qbit(2).data.toarray())
    # assert therm1 == therm2
