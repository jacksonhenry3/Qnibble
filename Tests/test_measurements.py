"""Tests that basis kets are working properly"""
from src import measurements as mes, density_matrix as DM


class TestMeasurements:
    """
    Testing that measurements produce the expected results.
    """

    def test_temp(self):
        """Tests the creation of a simple ket and validates its properties"""
        dm = DM.qbit(.25)
        assert mes.temp(dm) == .9102392266268373

        dm = DM.qbit(.4)
        assert mes.temp(dm) == 2.4663034623764326

    def test_temp_from_pop(self):
        dm = DM.qbit(.4)
        assert mes.temp(dm) == mes.temp_from_pop(.4)

    def test_pop_from_temp(self):
        dm = DM.qbit(.4)
        assert mes.pop_from_temp(mes.temp(dm)) == .4

    def test_D(self):
        dm_1, dm_2 = DM.qbit(.4), DM.qbit(.1)
        assert mes.D(dm_1, dm_1) == 0
        assert mes.D(dm_1, dm_2) == 0.3112386795830575

    def test_extractable_work(self):
        T = .4
        dm = DM.qbit(mes.pop_from_temp(T))
        assert mes.extractable_work(T, dm) == 0
        assert mes.extractable_work(T - .1, dm) > mes.extractable_work(T, dm)
        assert mes.extractable_work(T - .3, dm) > mes.extractable_work(T - .1, dm)

    def test_change_in_extractable_work(self):
        T_1, T_2 = .4, .2
        dm_1 = DM.qbit(mes.pop_from_temp(T_1))
        dm_2 = DM.qbit(mes.pop_from_temp(T_2))
        assert mes.change_in_extractable_work(T_1, dm_1, T_2, dm_2) == 0
        assert mes.change_in_extractable_work(T_1, dm_1, T_1, dm_1) == 0
        assert mes.change_in_extractable_work(T_1, dm_1, T_2, dm_1) == 0.02349421284993547
        assert mes.change_in_extractable_work(T_2, dm_1, T_1, dm_2) != 0

    def test_entropy(self):
        dm = DM.qbit(.3)
        assert mes.entropy(dm) == .6108643020548935

