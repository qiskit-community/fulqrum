# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test index ordering of FermionicOperator terms"""

from fulqrum import FermionicOperator


def test_index_ordering_simple1():
    """Test index ordering of single term FermionicOperator"""
    N = 5
    fo = FermionicOperator(N, [("-++-+", [3, 2, 3, 0, 0])])
    assert fo.operators == [("-", 0), ("+", 0), ("+", 2), ("-", 3), ("+", 3)]


def test_index_ordering_simple2():
    """Test index ordering of single term FermionicOperator"""
    N = 5
    fo = FermionicOperator(N, [("-++-+", [2, 2, 3, 0, 0])])
    assert fo.operators == [("-", 0), ("+", 0), ("-", 2), ("+", 2), ("+", 3)]


def test_index_ordering_preservation():
    """Test index ordering does not exchange terms with same indices"""
    N = 5
    fo = FermionicOperator(N, [("-++--++-", [2, 2, 0, 0, 3, 3, 1, 1])])
    assert fo.operators == [
        ("+", 0),
        ("-", 0),
        ("+", 1),
        ("-", 1),
        ("-", 2),
        ("+", 2),
        ("-", 3),
        ("+", 3),
    ]


def test_index_ordering_projectors1():
    """Test index ordering works with projectors"""
    fop = FermionicOperator(2, [("-0", [1, 0])])
    assert fop.operators == [("0", 0), ("-", 1)]
    assert fop.coeff == 1.0


def test_index_ordering_projectors2():
    """Test index ordering works with projectors"""
    fop = FermionicOperator(5, [("-0+", [4, 0, 3])])
    assert fop.operators == [("0", 0), ("+", 3), ("-", 4)]
    assert fop.coeff == -1.0


def test_index_ordering_projectors3():
    """Test index ordering works with projectors"""
    fop = FermionicOperator(5, [("-0+1", [3, 1, 0, 2])])
    assert fop.operators == [("+", 0), ("0", 1), ("1", 2), ("-", 3)]
    assert fop.coeff == -1.0
