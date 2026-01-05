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
"""Test string to vector conversion"""

import numpy as np
import qiskit_addon_fulqrum as fq


def test_bitset_int1():
    """Test bitset ladder int for different ladder widths"""
    bits = fq.Bitset("110101" * 20)
    inds = np.array([0, 17, 110, 115], dtype=np.uint32)
    for ladder_width in range(1, 5):
        assert bits.ladder_int(inds, ladder_width) == int(
            "".join([str(int(bits[kk])) for kk in inds[:ladder_width][::-1]]), 2
        )


def test_bitset_int2():
    """Test bitset ladder int for different ladder widths, #2"""
    bits = fq.Bitset("110101" * 20)
    inds = np.array([0, 1, 11, 119], dtype=np.uint32)
    for ladder_width in range(1, 5):
        assert bits.ladder_int(inds, ladder_width) == int(
            "".join([str(int(bits[kk])) for kk in inds[:ladder_width][::-1]]), 2
        )


def test_bitset_int3():
    """Test bitset ladder int for different indices at default width of 3"""
    bits = fq.Bitset("110101" * 20)
    for length in range(1, 5):
        inds = np.arange(1, length + 1, dtype=np.uint32)
        assert bits.ladder_int(inds, 3) == int(
            "".join([str(int(bits[kk])) for kk in inds[:3][::-1]]), 2
        )


def test_bitset_int4():
    """Test bitset ladder int for different indices at default width of 3, #2"""
    bits = fq.Bitset("110101" * 20)
    for length in range(1, 5):
        inds = np.arange(1, length + 1, dtype=np.uint32) + 18
        assert bits.ladder_int(inds, 3) == int(
            "".join([str(int(bits[kk])) for kk in inds[:3][::-1]]), 2
        )


def test_operator_ladder_int1():
    """Test ladder int works for various ladder widths"""
    op = fq.QubitOperator.from_label("I+-Z0X+-+Y")
    op.set_type(2)
    op.group_term_sort_by_ladder_int(5)
    assert op.ladder_ints()[0] == int("10101", 2)
    op.group_term_sort_by_ladder_int(4)
    assert op.ladder_ints()[0] == int("0101", 2)
    op.group_term_sort_by_ladder_int(3)
    assert op.ladder_ints()[0] == int("101", 2)
    op.group_term_sort_by_ladder_int(2)
    assert op.ladder_ints()[0] == int("01", 2)
    op.group_term_sort_by_ladder_int(1)
    assert op.ladder_ints()[0] == int("1", 2)
    # check default behavior
    op.group_term_sort_by_ladder_int()
    assert op.ladder_ints()[0] == 5


def test_operator_ladder_int2():
    """Test ladder int works multiple terms and no ladder ops"""
    op = fq.QubitOperator.from_label("I+-Z0X+-+Y") + fq.QubitOperator.from_label(
        "IXYI01IIXYZ"
    )
    op.set_type(2)
    op.group_term_sort_by_ladder_int()
    assert np.allclose(op.ladder_ints(), [np.iinfo(np.uint32).max, 5])


def test_operator_ladder_int3():
    """Test ladder int works if num. of ladder ops smaller than default of 3"""
    op = fq.QubitOperator.from_label("I+-Z0X+-+Y") + fq.QubitOperator.from_label(
        "IXYI01IIXYZ"
    )
    op.set_type(2)
    op.group_term_sort_by_ladder_int()
    assert np.allclose(op.ladder_ints(), [np.iinfo(np.uint32).max, 5])
