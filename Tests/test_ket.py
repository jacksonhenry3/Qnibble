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
        test_ket = ket.Ket(10, 5)
        assert test_ket.num == 10

    def test_equality(self):
        """tests equality comparisons between kets"""
        data = 10
        test_ket_1 = ket.Ket(data, 5)
        test_ket_2 = ket.Ket(data, 5)
        other_data = 11
        other_test_ket = ket.Ket(other_data, 5)
        assert test_ket_1 == test_ket_2
        assert test_ket_1 != other_test_ket

    def test_energy(self):
        """tests the extraction of energy value from kets"""
        data = 10
        test_ket = ket.Ket(data, 5)
        assert test_ket.energy == 2

    def test_ordering(self):
        """tests the ordering of kets based on energy"""
        data_1, data_2, data_3 = 8, 24, 27
        ket_1, ket_2, ket_3 = ket.Ket(data_1, 5), ket.Ket(data_2, 5), ket.Ket(data_3, 5)
        kets = [ket_1, ket_3, ket_2]
        assert ket_1 < ket_3
        assert ket_2 < ket_3
        assert ket_3 > ket_1
        assert sorted(kets) == [ket_1, ket_2, ket_3]

    def test_add(self):
        """tests the adition of kets to form larger kets"""
        data_1, data_2, data_3 = 8, 24, 27
        ket_1, ket_2, ket_3 = ket.Ket(data_1, 5), ket.Ket(data_2, 5), ket.Ket(data_3, 5)
        assert ket_1 + ket_2 != ket_2 + ket_1
        assert (ket_1 + ket_3).num == 283


class TestBasis:
    """ This is just a wrapper on tuple to keep track of basises better"""

    def test_creation(self):
        data_1, data_2 = 10, 11
        ket_1, ket_2 = ket.Ket(data_1, 5), ket.Ket(data_2, 5)
        basis_1, basis_2 = ket.Basis((ket_1, ket_2)), ket.Basis((ket_1, ket_1))
        assert basis_1 == (ket_1, ket_2)
        assert basis_2 != basis_1
        assert basis_2 != (ket_1, ket_2)


class TestFunctions:
    """Test the functions used to generate useful basises"""

    def test_canonical_basis(self):
        """tests that ket.canonical_basis generates the expected basis"""
        kets = ket.Ket(0,2), ket.Ket(1,2), ket.Ket(2,2), ket.Ket(3,2)
        assert ket.canonical_basis(2) == ket.Basis(kets)

    def test_energy_basis(self):
        """tests that ket.energy_basis generates the expected basis"""
        kets = ket.Ket(0,3), ket.Ket(1,3), ket.Ket(2,3), ket.Ket(4,3), ket.Ket(3,3), ket.Ket(5,3), ket.Ket(6,3), ket.Ket(7,3)
        assert ket.energy_basis(3) == ket.Basis(kets)
