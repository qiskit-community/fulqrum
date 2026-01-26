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
"""Test basic core functionality"""

from fulqrum import FermionicOperator


def test_empty_fermioperator():
    """Test empty FermionicOperator"""
    N = 5
    fo = FermionicOperator(N)
    assert fo.num_terms == 0


def test_id_fermioperator1():
    """Test identity FermionicOperator"""
    N = 5
    fo = FermionicOperator(N, [()])
    assert fo.num_terms == 1


def test_id_fermioperator2():
    """Test identity FermionicOperator"""
    N = 5
    fo = FermionicOperator(N, [[]])
    assert fo.num_terms == 1


def test_fermioperator_coeff():
    """Test setting coeff for FermionicOperator"""
    N = 5
    fo = FermionicOperator(N, [[1 + 2j]])
    assert fo.coeff == 1 + 2j


def test_fermioperator_single_term1():
    """Test single term FermionicOperator"""
    N = 5
    fo = FermionicOperator(N, [("+" * 5, range(5), 1)])
    assert fo.operators == [("+", kk) for kk in range(N)]


def test_fermioperator_single_term_weight():
    """Test single term FermionicOperator weight"""
    N = 5
    fo = FermionicOperator(N, [("+" * 5, range(5), 1)])
    assert fo.weights() == [5]


def test_fermioperator_inplace_addition():
    """Test FermionicOperator inplace addition"""
    N = 5
    fop = FermionicOperator(N)

    for kk in range(N):
        fop += FermionicOperator(N, [("-", kk, 1.0 / (N + kk))])

    for kk in range(N):
        assert fop[kk].operators == [("-", kk)]
        assert fop[kk].coeff == 1.0 / (N + kk)


def test_fermioperator_addition():
    """Test FermionicOperator addition"""
    N = 5
    fop = FermionicOperator(N)

    for kk in range(N):
        fop = fop + FermionicOperator(N, [("-", kk, 1.0 / (N + kk))])

    for kk in range(N):
        assert fop[kk].operators == [("-", kk)]
        assert fop[kk].coeff == 1.0 / (N + kk)


def test_fermioperator_single_term_repeat_indices1():
    """Test single term FermionicOperator repeated indices"""
    N = 5
    fo = FermionicOperator(N, [("+" * 5, [0] * 5, 1)])
    assert fo.operators == [("+", 0), ("+", 0), ("+", 0), ("+", 0), ("+", 0)]


def test_fermioperator_single_term_repeat_indices2():
    """Test single term FermionicOperator repeated indices"""
    N = 5
    fo = FermionicOperator(N, [("+--+-", [2, 1, 1, 0, 0], 1)])
    assert fo.operators == [("+", 0), ("-", 0), ("-", 1), ("-", 1), ("+", 2)]


def test_fermioperator_subtraction():
    """Test FermionicOperator subtraction"""
    N = 5
    fop1 = FermionicOperator(N, [("+", [0], 1)]) - FermionicOperator(N, [("-", [0], 2)])
    fop2 = FermionicOperator(N, [("+", [0], 1)]) + FermionicOperator(
        N, [("-", [0], -2)]
    )

    assert fop1[0].operators == fop2[0].operators
    assert fop1[0].coeff == fop2[0].coeff
    assert fop1[1].operators == fop2[1].operators
    assert fop1[1].coeff == fop2[1].coeff


def test_fermioperator_from_label():
    """Test Fermi from label construction"""
    fop1 = FermionicOperator.from_label(5, "+:0 1:2 -:3", -5 + 3j)
    fop2 = FermionicOperator(5, [("+1-", (0, 2, 3), -5 + 3j)])
    assert fop1.operators == fop2.operators
