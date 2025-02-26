# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test basic core functionality"""
import numpy as np
from fulqrum import QubitOperator


def test_unset_grouping():
    """Test that all groups are -1 for unsorted operator"""
    H = QubitOperator(2, [])
    for item in ["XY", "XI", "IY", "YY", "IZ", "II", "Z0"]:
        H += QubitOperator.from_label(item)
    ans = -1 * np.ones(7, dtype=np.int32)
    assert np.allclose(ans, H.groups())


def test_grouping1():
    """Test grouping"""
    H = QubitOperator(2, [])
    for item in ["XY", "XI", "IY", "YY", "IZ", "II", "Z0"]:
        H += QubitOperator.from_label(item)
    H.offdiag_term_grouping()
    # three diag terms, two terms of weight two, and one of each others
    ans = np.array([0, 0, 0, 1, 1, 2, 3], dtype=np.int32)
    assert np.allclose(ans, H.groups())


def test_grouping2():
    """Test grouping all diag terms gives all zero groups"""
    H = QubitOperator(3, [])
    for item in ["III", "ZZ1", "Z0Z", "IZI", "ZI0"]:
        H += QubitOperator.from_label(item)
    H.offdiag_term_grouping()
    ans = np.zeros(5, dtype=np.int32)
    assert np.allclose(ans, H.groups())


def test_grouping3():
    """Test all operators have same offdiag structure"""
    H = QubitOperator(4, [])
    for item in ["XIIY", "+ZZ-", "Y01X", "-00+"]:
        H += QubitOperator.from_label(item)
    H.offdiag_term_grouping()
    ans = np.ones(4, dtype=np.int32)
    assert np.allclose(ans, H.groups())
