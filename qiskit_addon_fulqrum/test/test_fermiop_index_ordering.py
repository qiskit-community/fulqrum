# This code is a Qiskit project.
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
"""Test index ordering of FermionicOperator terms"""

from qiskit_addon_fulqrum import FermionicOperator


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
