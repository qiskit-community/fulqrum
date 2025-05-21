# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test bitset object"""

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


def test_to_int():
    """Test Bitset to int (large)"""
    in_string = "101100" * 100
    bits = Bitset(in_string)
    out_int = bits.to_int()
    assert out_int == int(in_string, 2)
