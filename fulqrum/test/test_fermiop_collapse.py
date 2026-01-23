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

from fulqrum import FermionicOperator


def test_deflate_empty():
    """deflate indices works for empty operators"""
    fop = FermionicOperator(1, [])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_id():
    """deflate indices works for id operators"""
    fop = FermionicOperator(1, [[]])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == []


def test_deflate_single_elements1():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(1, [("-", [0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("-", 0)]


def test_deflate_single_elements2():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(2, [("+", [1])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("+", 1)]


def test_deflate_single_elements3():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(2, [("0", [1])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("0", 1)]


def test_deflate_single_elements4():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(2, [("1", [0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("1", 0)]


def test_deflate_pairs1():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("--", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs2():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("-+", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("0", 0)]


def test_deflate_pairs3():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("-0", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs4():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("-1", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("-", 0)]


def test_deflate_pairs5():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("+-", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("1", 0)]


def test_deflate_pairs6():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("++", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs7():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("+0", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("+", 0)]


def test_deflate_pairs8():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("+1", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs9():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("0-", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("-", 0)]


def test_deflate_pairs10():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("0+", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs11():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("00", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("0", 0)]


def test_deflate_pairs12():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("01", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs13():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("1-", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs14():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("1+", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("+", 0)]


def test_deflate_pairs15():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("10", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs16():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [("11", [0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("1", 0)]


def test_deflate_three1():
    """deflate 3 elements"""
    fop = FermionicOperator(1, [("+-+", [0, 0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("+", 0)]


def test_deflate_three2():
    """deflate 3 elements"""
    fop = FermionicOperator(1, [("11-", [0, 0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_three3():
    """deflate 3 elements"""
    fop = FermionicOperator(1, [("-+-", [0, 0, 0])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("-", 0)]


def test_deflate_two_pair1():
    """deflate two pairs of elements"""
    fop = FermionicOperator(2, [("+--+", [0, 0, 1, 1])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("1", 0), ("0", 1)]


def test_deflate_two_pair2():
    """deflate two pairs of elements"""
    fop = FermionicOperator(2, [("+-++", [0, 0, 1, 1])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == None


def test_deflate_several():
    """deflate several elements"""
    fop = FermionicOperator(2, [("+-+-+", [0, 0, 1, 1, 1])])
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.operators == [("1", 0), ("+", 1)]


def test_deflate_multi_term():
    """deflate multiple terms"""
    fop = FermionicOperator(3, [("+--+", [0, 0, 1, 2])])
    fop += FermionicOperator(3, [("-+", [1, 1])])
    fop += FermionicOperator(3, [("+--++", [0, 0, 1, 2, 2])])  # term is zero
    fop_deflate = fop.deflate_repeated_indices()
    assert fop_deflate.num_terms == 2  # since last term is zero
    assert fop_deflate[0].operators == [("1", 0), ("-", 1), ("+", 2)]
    assert fop_deflate[1].operators == [("0", 1)]
