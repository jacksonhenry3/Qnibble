"""Tests that density matrices are working properly"""
import numpy as np
import pytest
from src import density_matrix as dm, ket as ket
import scipy as sp


class TestKet:
    """
    Testing the functionality of density matrices.
    The desired functionality is very specific compared to a general ket.
    """

    def test_create(self):
        """Tests the creation of a simple ket and validates its properties"""
        dm.DensityMatrix(matrix=np.identity(1), basis=ket.Basis([ket.Ket([0])]))

    def test_equality(self):
        """Tests the creation of a simple ket and validates its properties"""
        data_1, data_2, data_3 = np.identity(3), np.identity(3), np.identity(2)
        basis = ket.Basis([ket.Ket([0])])
        assert dm.DensityMatrix(data_1, basis) == dm.DensityMatrix(data_1, basis)
        assert dm.DensityMatrix(data_1, basis) == dm.DensityMatrix(data_2, basis)
        assert dm.DensityMatrix(data_1, basis) != dm.DensityMatrix(data_3, basis)

    def test_add(self):
        """Test the addition of multiple density matrices"""
        data_1, data_2, data_3 = np.identity(4), np.ones((4, 4)), np.zeros((4, 4))
        basis_1 = ket.Basis([ket.Ket('00'), ket.Ket('01'), ket.Ket('10'), ket.Ket('11')])
        dm_1_1, dm_1_2, dm_1_3 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1), dm.DensityMatrix(data_3, basis_1)
        assert dm_1_1 + dm_1_3 == dm_1_1
        assert np.array_equal((dm_1_1 + dm_1_2).data, data_1 + data_2)
        assert (dm_1_2 + dm_1_2).basis == basis_1

        basis_2 = ket.Basis([ket.Ket('01'), ket.Ket('00'), ket.Ket('10'), ket.Ket('11')])
        dm_2_1, dm_2_2 = dm.DensityMatrix(data_1, basis_2), dm.DensityMatrix(data_2, basis_2)
        with pytest.raises(AssertionError):
            assert dm_1_1 + dm_2_2 == dm_1_1

    def test_tensor(self):
        """Test the tensor product of density matrices"""
        data_1, data_2 = np.identity(2), np.ones((2, 2))
        basis_1 = ket.Basis([ket.Ket('0'), ket.Ket('1')])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        result = dm_1.tensor(dm_2)
        assert np.array_equal(result.data, sp.kron(data_1, data_2))

    def test_multiply(self):
        data_1, data_2 = np.identity(2), np.ones((2, 2))
        basis_1 = ket.Basis([ket.Ket('0'), ket.Ket('1')])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm_1 * dm_2 == dm_2
        assert np.array_equal((dm_1 * 3).data, 3 * data_1)
        assert np.array_equal((3 * dm_2).data, 3 * data_2)

    def test_pow(self):
        data_1, data_2 = np.identity(2), np.ones((2, 2))
        basis_1 = ket.Basis([ket.Ket('0'), ket.Ket('1')])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm_1 ** 3 == dm_1
        assert dm_2 ** 3 == 4 * dm_2

    def test_H(self):
        """test the hermitian conjugate"""
        data_1, data_2 = np.identity(2), np.array([[1j, 0], [0, 0]])
        basis_1 = ket.Basis([ket.Ket('0'), ket.Ket('1')])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm_1.H.H == dm_1
        assert np.array_equal(dm_2.H.data, np.conjugate(np.transpose(data_2)))

    def test_dm_exp(self):
        data_1, data_2 = np.identity(4), np.ones((4, 4))
        basis_1 = ket.Basis([ket.Ket('00'), ket.Ket('01'), ket.Ket('10'), ket.Ket('11')])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert np.array_equal(dm.dm_exp(dm_1).data, sp.linalg.expm(dm_1.data))
        assert not np.array_equal(dm.dm_exp(dm_1).data, sp.linalg.expm(dm_2.data))

    def test_dm_log(self):
        data_1, data_2 = np.identity(4), np.ones((4, 4))
        basis_1 = ket.Basis([ket.Ket('00'), ket.Ket('01'), ket.Ket('10'), ket.Ket('11')])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert np.array_equal(dm.dm_log(dm_1).data, sp.linalg.logm(dm_1.data))
        assert not np.array_equal(dm.dm_log(dm_1).data, sp.linalg.logm(dm_2.data))
        assert dm.dm_log(dm.dm_exp(dm_1)) == dm_1

    def test_dm_trace(self):
        data_1, data_2 = np.identity(4), np.ones((4, 4))
        basis_1 = ket.Basis([ket.Ket('00'), ket.Ket('01'), ket.Ket('10'), ket.Ket('11')])
        dm_1, dm_2 = dm.DensityMatrix(data_1, basis_1), dm.DensityMatrix(data_2, basis_1)
        assert dm.dm_trace(dm_1) == np.trace(data_1)
        assert dm.dm_trace(dm_2) == np.trace(data_2)

