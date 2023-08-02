"""Tests that density matrices are working properly"""

import pytest

import random_unitary
from src import density_matrix as dm, ket as ket
import src.setup as setup
from src.setup import SPARSE_TYPE, xp

import numpy as np
import scipy as sp
from scipy import sparse
import scipy.linalg as sp_dense


def identity(n):
    return SPARSE_TYPE(sparse.identity(n, dtype=complex))


class TestKet:
    """
    Testing the functionality of density matrices.
    The desired functionality is very specific compared to a general ket.
    """

    def test_create(self):
        """Tests the creation of a simple ket and validates its properties"""
        dm.DensityMatrix(matrix=identity(1), basis=ket.Basis([ket.Ket(0, 1)]))

    def test_equality(self):
        """Tests the creation of a simple ket and validates its properties"""
        data_1, data_2, data_3 = identity(3), identity(3), identity(2)
        basis = ket.Basis([ket.Ket(0, 1)])
        assert dm.DensityMatrix(data_1, basis) == dm.DensityMatrix(data_1, basis)
        assert dm.DensityMatrix(data_1, basis) == dm.DensityMatrix(data_2, basis)
        assert dm.DensityMatrix(data_1, basis) != dm.DensityMatrix(data_3, basis)

    def test_add(self):
        """Test the addition of multiple density matrices"""
        data_1, data_2, data_3 = identity(4), dm.SPARSE_TYPE(np.ones((4, 4))), dm.SPARSE_TYPE(np.zeros((4, 4)))
        basis_1 = ket.Basis([ket.Ket(0, 2), ket.Ket(1, 2), ket.Ket(2, 2), ket.Ket(3, 2)])
        dm_1_1, dm_1_2, dm_1_3 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1), dm.DensityMatrix(data_3, basis_1)
        assert dm_1_1 + dm_1_3 == dm_1_1
        assert ((dm_1_1 + dm_1_2).data != data_1 + data_2).nnz == 0
        assert (dm_1_2 + dm_1_2).basis == basis_1

        basis_2 = ket.Basis([ket.Ket(1, 2), ket.Ket(0, 2), ket.Ket(2, 2), ket.Ket(3, 2)])
        dm_2_1, dm_2_2 = dm.DensityMatrix(data_1, basis_2), dm.DensityMatrix(data_2, basis_2)
        with pytest.raises(AssertionError):
            assert dm_1_1 + dm_2_2 == dm_1_1

    def test_tensor(self):
        """Test the tensor product of density matrices"""
        data_1, data_2 = identity(2), dm.SPARSE_TYPE(np.ones((2, 2)))
        basis_1 = ket.Basis([ket.Ket(0, 1), ket.Ket(1, 1)])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        result = dm_1.tensor(dm_2)
        assert (result.data != sparse.kron(data_1, data_2)).nnz == 0

    def test_multiply(self):
        data_1, data_2 = identity(2), dm.SPARSE_TYPE(np.ones((2, 2)))
        basis_1 = ket.Basis([ket.Ket(0, 1), ket.Ket(1, 1)])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm_1 * dm_2 == dm_2
        assert ((dm_1 * 3).data != 3 * data_1).nnz == 0
        assert ((3 * dm_2).data != 3 * data_2).nnz == 0

    def test_pow(self):
        data_1, data_2 = identity(2), dm.SPARSE_TYPE(np.ones((2, 2)))
        basis_1 = ket.Basis([ket.Ket(0, 1), ket.Ket(1, 1)])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm_1 ** 3 == dm_1
        assert dm_2 ** 3 == 4 * dm_2

    def test_H(self):
        """test the hermitian conjugate"""
        data_1, data_2 = identity(2), SPARSE_TYPE([[1j, 0], [0, 0]])
        basis_1 = ket.Basis([ket.Ket(0, 1), ket.Ket(1, 1)])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm_1.H.H == dm_1
        assert (dm_2.H.data != xp.conjugate(xp.transpose(data_2))).nnz == 0

    def test_dm_exp(self):
        data_1, data_2 = identity(4), dm.SPARSE_TYPE(np.ones((4, 4)))
        basis_1 = ket.Basis([ket.Ket(0, 2), ket.Ket(1, 2), ket.Ket(2, 2), ket.Ket(3, 2)])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert (dm.dm_exp(dm_1).data != sp.linalg.expm(dm_1.data)).nnz == 0
        assert not (dm.dm_exp(dm_1).data != sp.linalg.expm(dm_2.data)).nnz == 0

    def test_dm_log(self):
        data_1, data_2 = identity(4), dm.SPARSE_TYPE(np.ones((4, 4)))
        basis_1 = ket.Basis([ket.Ket(0, 2), ket.Ket(1, 2), ket.Ket(2, 2), ket.Ket(3, 2)])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert np.allclose(dm.dm_log(dm_1).data.toarray(), sp_dense.logm(dm_1.data.toarray()))
        assert not np.allclose(dm.dm_log(dm_1).data.toarray(), sp_dense.logm(dm_2.data.toarray()))
        assert dm.dm_log(dm.dm_exp(dm_1)) == dm_1

    def test_dm_trace(self):
        data_1, data_2 = identity(4), dm.SPARSE_TYPE(np.ones((4, 4)))
        basis_1 = ket.Basis([ket.Ket(0, 2), ket.Ket(1, 2), ket.Ket(2, 2), ket.Ket(3, 2)])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm.dm_trace(dm_1) == np.trace(data_1.toarray())
        assert dm.dm_trace(dm_2) == np.trace(data_2.toarray())

    def test_dm_partial_trace(self):
        dm_1, dm_2 = dm.n_thermal_qbits([.1, .2]), dm.n_thermal_qbits([.3, .02])
        U1, U2 = random_unitary.random_energy_preserving_unitary(2), random_unitary.random_energy_preserving_unitary(2)
        dm_1 = U1 * dm_1 * U1.H
        dm_2 = U1 * dm_2 * U1.H
        total = dm_1.tensor(dm_2)
        res = total.ptrace([0, 1])
        assert dm_2 == res
