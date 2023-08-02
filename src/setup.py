import numpy as xp

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

SPARSE_TYPE = sp.sparse.csc_matrix

using_gpu = False


def use_gpu():
    """ Call this function before importing anything else if you want to run using CUPY on the GPU"""
    global xp
    global sp
    global SPARSE_TYPE
    global using_gpu
    using_gpu = True
    import cupy as xp
    import cupyx.scipy as sp

    import cupyx.scipy.sparse
    import cupyx.scipy.linalg
    import cupyx.scipy.sparse.linalg
    SPARSE_TYPE = sp.sparse.csr_matrix
