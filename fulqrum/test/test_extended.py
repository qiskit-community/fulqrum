# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test basic core functionality"""
import numpy as np
from fulqrum import QubitOperator


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
