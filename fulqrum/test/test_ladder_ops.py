# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test string to vector conversion"""

import numpy as np
import fulqrum as fq


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
        assert bits.ladder_int(inds) == int(
            "".join([str(int(bits[kk])) for kk in inds[:3][::-1]]), 2
        )


def test_bitset_int4():
    """Test bitset ladder int for different indices at default width of 3, #2"""
    bits = fq.Bitset("110101" * 20)
    for length in range(1, 5):
        inds = np.arange(1, length + 1, dtype=np.uint32) + 18
        assert bits.ladder_int(inds) == int(
            "".join([str(int(bits[kk])) for kk in inds[:3][::-1]]), 2
        )


def test_operator_ladder_int1():
    """Test ladder int works for various ladder widths"""
    op = fq.QubitOperator.from_label("I+-Z0X+-+Y")
    assert op.ladder_ints(5)[0] == int("10101", 2)
    assert op.ladder_ints(4)[0] == int("0101", 2)
    assert op.ladder_ints(3)[0] == int("101", 2)
    assert op.ladder_ints(2)[0] == int("01", 2)
    assert op.ladder_ints(1)[0] == int("1", 2)
    # check default behavior
    assert op.ladder_ints()[0] == 5


def test_operator_ladder_int2():
    """Test ladder int works multiple terms and no ladder ops"""
    op = fq.QubitOperator.from_label("I+-Z0X+-+Y") + fq.QubitOperator.from_label(
        "IXYI01IIXYZ"
    )
    assert np.allclose(op.ladder_ints(), [5, np.iinfo(np.uint32).max])


def test_operator_ladder_int3():
    """Test ladder int works if num. of ladder ops smalelr than default of 3"""
    op = fq.QubitOperator.from_label("I+-Z0X+-+Y") + fq.QubitOperator.from_label(
        "IXYI01IIXYZ"
    )
    assert np.allclose(op.ladder_ints(), [5, np.iinfo(np.uint32).max])
