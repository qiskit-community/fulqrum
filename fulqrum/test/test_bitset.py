# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=no-name-in-module
"""Test bitset object"""
from fulqrum import QubitOperator
from fulqrum.core import Bitset


def test_to_string1():
    """Test simple string to Bitset and back to string (small)"""
    in_string = "001101"
    bits = Bitset(in_string)
    out_string = bits.to_string()
    assert in_string == out_string


def test_to_string2():
    """Test simple string to Bitset and back to string (large)"""
    in_string = "101100" * 100
    bits = Bitset(in_string)
    out_string = bits.to_string()
    assert in_string == out_string


def test_to_int1():
    """Test Bitset to int (large)"""
    in_string = "101100" * 100
    bits = Bitset(in_string)
    out_int = bits.to_int()
    assert out_int == int(in_string, 2)


def test_to_int2():
    """Test two different len bitsets that equal same int"""
    string1 = "101110"
    bits1 = Bitset(string1)

    string2 = "0101110"
    bits2 = Bitset(string2)
    assert bits1.to_int() == bits2.to_int()


def test_equality():
    """Test Bitset equality"""
    string1 = "101110"
    bits1 = Bitset(string1)
    bits2 = Bitset(string1)

    assert bits1 == bits2


def test_inequality1():
    """Test Bitset inequality, same length"""
    string1 = "101110"
    bits1 = Bitset(string1)

    string2 = "101111"
    bits2 = Bitset(string2)

    assert bits1 != bits2


def test_inequality2():
    """Test Bitset inequality, different lengths"""
    string1 = "101110"
    bits1 = Bitset(string1)

    string2 = "0101110"
    bits2 = Bitset(string2)

    assert bits1 != bits2


def test_offdiag_flip1():
    """Test flipping off-diagonal bits for a given operator"""
    N = 100
    bits = Bitset("0" * N)
    inds = [0, 5, 17, 77]
    op_str = "X" * len(inds)
    op = QubitOperator(N, [(op_str, inds, 1.0)])
    new_bits = bits.offdiag_flip(op)
    new_str = new_bits.to_string()
    for ind in inds:
        assert new_str[N - ind - 1] == "1"


def test_offdiag_flip2():
    """Test flipping off-diagonal bits for a given operator"""
    N = 100
    bits = Bitset("1" * N)
    inds = [1, 87, 88, 91]
    op_str = "+" * len(inds)
    op = QubitOperator(N, [(op_str, inds, 1.0)])
    new_bits = bits.offdiag_flip(op)
    new_str = new_bits.to_string()
    for ind in inds:
        assert new_str[N - ind - 1] == "0"
