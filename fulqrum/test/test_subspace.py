# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test string to vector conversion"""

import numpy as np
from fulqrum import Subspace


dic = {"01010": 1, "10101": 1, "11111": 1, "11100": 1}
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
    V = Subspace(dic)
    # Order is only guarenteed up to first bit
    assert V[0].to_string()[-1] == "0"
    assert V[1].to_string()[-1] == "1"
    assert V[2].to_string()[-1] == "1"
    assert V[3].to_string()[-1] == "0"

def test_subspace_vector_order1():
    V = Subspace(dic2)
    # Order is only guarenteed up to first bit
    assert V[0].to_string()[-1] == "0"
    assert V[1].to_string()[-1] == "1"
    assert V[2].to_string()[-1] == "1"
    assert V[3].to_string()[-1] == "0"
    assert V[4].to_string()[-1] == "1"
    assert V[5].to_string()[-1] == "0"
    assert V[6].to_string()[-1] == "1"

def test_subspace_vector_order2():
    V = Subspace(dic)
    assert list(V.to_dict().keys()) == ["01010", "10101", "11111", "11100"]

def test_subspace_vector_order3():
    V = Subspace(dic2)
    assert list(V.to_dict().keys()) == [
        "01010",
        "10101",
        "11111",
        "11100",
        "01111",
        "00100",
        "10111"
    ]
