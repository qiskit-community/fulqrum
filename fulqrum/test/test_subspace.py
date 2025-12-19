# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test string to vector conversion"""

import pytest
import numpy as np
import pytest
from fulqrum import Subspace
from fulqrum.exceptions import FulqrumError


dic = {"01010": 1, "10101": 1, "11111": 1, "11100": 1} # sorted keys: "01010","10101","11100","11111"
dic = [list(dic.keys())]
dic2 = {
    "01010": 1,
    "10101": 1,
    "11111": 1,
    "11100": 1,
    "01111": 1,
    "00100": 1,
    "10111": 1,
} # sorted keys: "00100","01010","01111","10111","11100","11111"
dic2 = [list(dic2.keys())]


def test_subspace_vector_order0():
    V = Subspace(dic)
    # Order is only guarenteed up to first bit
    assert V[0].to_string()[-1] == "0"
    assert V[1].to_string()[-1] == "1"
    assert V[2].to_string()[-1] == "0"
    assert V[3].to_string()[-1] == "1"


def test_subspace_vector_order1():
    V = Subspace(dic2)
    # Order is only guarenteed up to first bit
    assert V[0].to_string()[-1] == "0"
    assert V[1].to_string()[-1] == "0"
    assert V[2].to_string()[-1] == "1"
    assert V[3].to_string()[-1] == "1"
    assert V[4].to_string()[-1] == "1"
    assert V[5].to_string()[-1] == "0"
    assert V[6].to_string()[-1] == "1"


def test_subspace_vector_order2():
    V = Subspace(dic)
    assert list(V.to_dict().keys()) == ["01010", "10101", "11100", "11111"]


def test_subspace_vector_order3():
    V = Subspace(dic2)
    assert list(V.to_dict().keys()) == [
        "00100",
        "01010",
        "01111",
        "10101",
        "10111",
        "11100",
        "11111",
    ]


def test_get_n_th_bitstring():
    """Test get_n_th_bitstring() method returns the correct bitstring
    Both Python and Fulqrum's emhash8::HashMap dictionaries retain
    the insertion order, and this test checks in that order.
    """
    V = Subspace(dic)
    assert V.get_n_th_bitstring(0) == "01010"
    assert V.get_n_th_bitstring(1) == "10101"
    assert V.get_n_th_bitstring(2) == "11100"
    assert V.get_n_th_bitstring(3) == "11111"


def test_max_num_qubits():
    bitstrings = [["0" * (2 ** 16), "1" * (2 ** 16)]]
    S = Subspace(bitstrings)
    
    assert S

    bitstrings = [["0" * (2 ** 16 + 1), "1" * (2 ** 16 + 1)]]

    with pytest.raises(ValueError):
        S = Subspace(bitstrings)


def test_get_orbital_occupancies():
    # sorted bitstrings as Subspace input as Subspace sorts them
    # if this is not sorted here, ``probs`` array may have mismatched order.
    bitstrings = [['011101', '101011', '110110']] # bit order: bN ... b0, aN ... a0
    probs = np.array([0.50, 0.30, 0.20])
    n_spatial_orb = len(bitstrings[0][0]) // 2
    S = Subspace(bitstrings)

    # occupancies orrder: ([a0, ..., aN], [b0, ..., bN])
    occs_a , occs_b = S.get_orbital_occupancies(probs=probs, norb=n_spatial_orb)

    expected_occs_a = np.array([0.80, 0.50, 0.70])
    expected_occs_b = np.array([0.80, 0.70, 0.50])

    np.testing.assert_allclose(occs_a, expected_occs_a)
    np.testing.assert_allclose(occs_b, expected_occs_b)

    bitstrings = [["00010001", "00101000", "10000100"]] # bit order: bN ... b0, aN ... a0
    probs = np.array([0.40, 0.50, 0.10])
    n_spatial_orb = len(bitstrings[0][0]) // 2
    S = Subspace(bitstrings)

    occs_a , occs_b = S.get_orbital_occupancies(probs=probs, norb=n_spatial_orb)

    expected_occs_a = np.array([0.40, 0.0, 0.10, 0.50])
    expected_occs_b = np.array([0.40, 0.50, 0.0, 0.10])

    np.testing.assert_allclose(occs_a, expected_occs_a)
    np.testing.assert_allclose(occs_b, expected_occs_b)
