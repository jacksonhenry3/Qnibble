"""Tests that basis kets are working properly"""
from src import ket as ket
import numpy as np

class TestKet:
    """
    Testing the functionality of kets.
    The desired functionality is very specific compared to a general ket.
    """

    def test_create(self):
        """Tests the creation of a simple ket and validates its properties"""
        test_ket = ket.Ket([0, 1, 0, 1, 0])
        assert np.array_equal(test_ket.data,np.array([0, 1, 0, 1, 0]))
        assert test_ket.num == 10

    def test_iter(self):
        """Tests the iterator properties of a ket"""
        data = [0, 1, 0, 1, 0]
        test_ket = ket.Ket(data)
        for i, b in enumerate(test_ket):
            assert data[i] == b

    def test_equality(self):
        """tests equality comparisons between kets"""
        data = [0, 1, 0, 1, 0]
        test_ket_1 = ket.Ket(data)
        test_ket_2 = ket.Ket(data)
        other_data = [0, 1, 0, 1, 1]
        other_test_ket = ket.Ket(other_data)
        assert test_ket_1 == test_ket_2
        assert test_ket_1 != other_test_ket

    def test_energy(self):
        """tests the extraction of energy value from kets"""
        data = [0, 1, 0, 1, 0]
        test_ket = ket.Ket(data)
        assert test_ket.energy == 2

    def test_ordering(self):
        """tests the ordering of kets based on energy"""
        data_1, data_2, data_3 = [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 1, 1]
        ket_1, ket_2, ket_3 = ket.Ket(data_1), ket.Ket(data_2), ket.Ket(data_3)
        kets = [ket_1, ket_3, ket_2]
        assert ket_1 < ket_3
        assert ket_2 < ket_3
        assert ket_3 > ket_1
        assert sorted(kets) == [ket_1, ket_2, ket_3]

    def test_add(self):
        """tests the adition of kets to form larger kets"""
        data_1, data_2, data_3 = [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 1, 1]
        ket_1, ket_2, ket_3 = ket.Ket(data_1), ket.Ket(data_2), ket.Ket(data_3)
        assert ket_1 + ket_2 == ket.Ket(data_1 + data_2)
        assert ket_1 + ket_2 + ket_3 == ket.Ket(data_1 + data_2 + data_3)


class TestBasis:
    """ This is just a wrapper on tuple to keep track of basises better"""

    def test_creation(self):
        data_1, data_2 = [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]
        ket_1, ket_2 = ket.Ket(data_1), ket.Ket(data_2)
        basis_1, basis_2 = ket.Basis((ket_1, ket_2)), ket.Basis((ket_1, ket_1))
        assert basis_1 == (ket_1, ket_2)
        assert basis_2 != basis_1
        assert basis_2 != (ket_1, ket_2)


class TestFunctions:
    """Test the functions used to generate useful basises"""

    def test_canonical_basis(self):
        """tests that ket.canonical_basis generates the expected basis"""
        kets = ket.Ket(list("00")), ket.Ket(list("01")), ket.Ket(list("10")), ket.Ket(list("11"))
        assert ket.canonical_basis(2) == ket.Basis(kets)

    def test_energy_basis(self):
        """tests that ket.energy_basis generates the expected basis"""
        kets = ket.Ket(list("000")), ket.Ket(list("001")), ket.Ket(list("010")), ket.Ket(list("100")), ket.Ket(list("011")), ket.Ket(list("101")), ket.Ket(list("110")), ket.Ket(list("111"))
        assert ket.energy_basis(3) == ket.Basis(kets)
