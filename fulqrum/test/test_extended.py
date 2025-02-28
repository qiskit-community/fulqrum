# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test extended operator functionality"""
import numpy as np
from fulqrum import QubitOperator

from .oper_funcs import nonzero_extended_value_wrapper


def test_extended1():
    """Test all id ops"""
    H = QubitOperator(2, [])
    for item in ["II", "II"]:
        H += QubitOperator.from_label(item)
    ans = np.array([0, 0], dtype=np.int32)
    assert np.allclose(H.extended(), ans)


def test_extended2():
    """Test mixed ops"""
    H = QubitOperator.from_label("-III") + QubitOperator.from_label("ZIII")
    ans = np.array([1, 0], dtype=np.int32)
    assert np.allclose(H.extended(), ans)


def test_extended3():
    """Test mixed ops"""
    H = QubitOperator.from_label("ZIII") + QubitOperator.from_label("-III")
    ans = np.array([0, 1], dtype=np.int32)
    assert np.allclose(H.extended(), ans)


def test_extended3():
    """Test mixed ops"""
    H = QubitOperator(4, [])
    for item in ["IIXY", "XY10", "ZIIZ", "III+"]:
        H += QubitOperator.from_label(item)
    ans = np.array([0, 1, 0, 1], dtype=np.int32)
    assert np.allclose(H.extended(), ans)


def test_extended_nonzero1():
    """Test extended nonzero check 1"""
    H = QubitOperator.from_label("-III")
    # Putting 0 on a - operator gives a zero
    bits = np.array([0, 0, 0, 0], dtype=np.uint8)
    assert nonzero_extended_value_wrapper(H, bits)


def test_extended_nonzero2():
    """Test extended nonzero check 2"""
    H = QubitOperator.from_label("-III")
    # Putting 1 on a - operator gives a zero
    bits = np.array([1, 0, 0, 0], dtype=np.uint8)
    assert not nonzero_extended_value_wrapper(H, bits)


def test_extended_nonzero3():
    """Test extended nonzero check 3"""
    H = QubitOperator.from_label("-III")
    # Putting 1 on a - operator gives a zero, others dont matter
    bits = np.array([1, 1, 0, 1], dtype=np.uint8)
    assert not nonzero_extended_value_wrapper(H, bits)


def test_extended_nonzero4():
    """Test extended nonzero check 4"""
    H = QubitOperator.from_label("-10+")
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert nonzero_extended_value_wrapper(H, bits)


def test_extended_nonzero5():
    """Test extended nonzero check 5"""
    H = QubitOperator.from_label("-10+")
    bits = np.array([0, 1, 1, 1], dtype=np.uint8)
    assert not nonzero_extended_value_wrapper(H, bits)


def test_extended_nonzero6():
    """Test extended nonzero check 6"""
    H = QubitOperator.from_label("-10+")
    bits = np.array([0, 1, 0, 0], dtype=np.uint8)
    assert not nonzero_extended_value_wrapper(H, bits)


def test_extended_nonzero7():
    """Test extended nonzero check 7"""
    for label in ["0000", "----"]:
        H = QubitOperator.from_label(label)
        bits = np.array([0, 0, 0, 0], dtype=np.uint8)
        assert nonzero_extended_value_wrapper(H, bits)


def test_extended_nonzero8():
    """Test extended nonzero check 8"""
    for label in ["1111", "++++"]:
        H = QubitOperator.from_label(label)
        bits = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert nonzero_extended_value_wrapper(H, bits)
