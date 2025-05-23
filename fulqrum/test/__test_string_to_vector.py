# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test string to vector conversion"""

import numpy as np
from fulqrum.test.string_funcs import str_to_vec_test


def test_str_to_vec1():
    """Test simple string to vec conversion"""
    string = "11111"
    res = str_to_vec_test(string)
    ans = np.array([int(kk) for kk in string], dtype=np.uintp)
    assert np.allclose(res, ans)
    assert res.dtype == ans.dtype  # check for correct dtype once


def test_str_to_vec2():
    """Test simple string to vec conversion"""
    string = "010101"
    res = str_to_vec_test(string)
    ans = np.array([int(kk) for kk in string], dtype=np.uintp)
    assert np.allclose(res, ans)


def test_str_to_vec3():
    """Test simple string to vec conversion"""
    string = "01011110101101"
    res = str_to_vec_test(string)
    ans = np.array([int(kk) for kk in string], dtype=np.uintp)
    assert np.allclose(res, ans)
    assert res.shape[0] == len(string)
