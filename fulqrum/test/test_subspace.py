# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test string to vector conversion"""

import numpy as np
from fulqrum import Subspace


dic = {"01010": 1, "10101": 1, "111111": 1, "11100": 1}
dic2 = {
    "01010": 1,
    "10101": 1,
    "11111": 1,
    "11100": 1,
    "01111": 1,
    "00100": 1,
    "10111": 1,
}


def test_subspace_vector_order1():
    V = Subspace(dic, bin_width=1)
    # Order is only guarenteed up to first bit
    assert V[0][-1] == 0
    assert V[1][-1] == 0
    assert V[2][-1] == 1
    assert V[3][-1] == 1


def test_subspace_vector_order2():
    V = Subspace(dic, bin_width=2)
    assert np.allclose(V[0], np.array([1, 1, 1, 0, 0], dtype=np.uint8))
    assert np.allclose(V[1], np.array([1, 0, 1, 0, 1], dtype=np.uint8))
    assert np.allclose(V[2], np.array([0, 1, 0, 1, 0], dtype=np.uint8))
    assert np.allclose(V[3], np.array([1, 1, 1, 1, 1], dtype=np.uint8))


def test_subspace_vector_order3():
    V = Subspace(dic, bin_width=3)
    assert np.allclose(V[0], np.array([0, 1, 0, 1, 0], dtype=np.uint8))
    assert np.allclose(V[1], np.array([1, 1, 1, 0, 0], dtype=np.uint8))
    assert np.allclose(V[2], np.array([1, 0, 1, 0, 1], dtype=np.uint8))
    assert np.allclose(V[3], np.array([1, 1, 1, 1, 1], dtype=np.uint8))


def test_subspace_vector_order4():
    V = Subspace(dic, bin_width=4)
    assert np.allclose(V[0], np.array([1, 0, 1, 0, 1], dtype=np.uint8))
    assert np.allclose(V[1], np.array([0, 1, 0, 1, 0], dtype=np.uint8))
    assert np.allclose(V[2], np.array([1, 1, 1, 0, 0], dtype=np.uint8))
    assert np.allclose(V[3], np.array([1, 1, 1, 1, 1], dtype=np.uint8))


def test_subspace_vector_order5():
    V = Subspace(dic, bin_width=5)
    assert np.allclose(V[0], np.array([0, 1, 0, 1, 0], dtype=np.uint8))
    assert np.allclose(V[1], np.array([1, 0, 1, 0, 1], dtype=np.uint8))
    assert np.allclose(V[2], np.array([1, 1, 1, 0, 0], dtype=np.uint8))
    assert np.allclose(V[3], np.array([1, 1, 1, 1, 1], dtype=np.uint8))


def test_subspace_bin_counts1():
    V = Subspace(dic2, bin_width=1)
    assert np.allclose(V.bin_sizes(), np.array([3, 4], dtype=np.uintp))


def test_subspace_bin_counts2():
    V = Subspace(dic2, bin_width=2)
    assert np.allclose(V.bin_sizes(), np.array([2, 1, 1, 3], dtype=np.uintp))


def test_subspace_bin_counts3():
    V = Subspace(dic2, bin_width=3)
    assert np.allclose(
        V.bin_sizes(), np.array([0, 0, 1, 0, 2, 1, 0, 3], dtype=np.uintp)
    )


def test_subspace_bin_counts5():
    V = Subspace(dic2, bin_width=5)
    int_values = [int(kk, 2) for kk in dic2.keys()]
    res = np.zeros(2**5, dtype=np.uintp)
    res[int_values] = 1
    assert np.allclose(V.bin_sizes(), res)


def test_subspace_auto_binwidth():
    temp_dic = {"010010": 10, "100101": 5, "1110111": 1, "101100": 55}
    V = Subspace(temp_dic)
    assert V.bin_width == 6
