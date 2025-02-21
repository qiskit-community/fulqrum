# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test string to vector conversion"""

import numpy as np
from fulqrum import QubitOperator
from fulqrum.test.string_funcs import find_col_vec_test


def test_find_col1():
    """Test simple string to vec conversion"""
    string = "11"
    op = QubitOperator.from_label("XX")
    res = find_col_vec_test(op, string)
    ans = np.array([0, 0], dtype=np.uintp)
    assert np.allclose(res, ans)
