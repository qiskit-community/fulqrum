# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test index ordering of FermionicOperator terms"""

from fulqrum import FermionicOperator


def test_deflate_empty():
    """deflate indices works for empty operators"""
    fop = FermionicOperator(1, [ ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_id():
    """deflate indices works for id operators"""
    fop = FermionicOperator(1, [[]])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == []


def test_deflate_single_elements1():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(1, [ ('-', [0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('-', 0)]


def test_deflate_single_elements2():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(2, [ ('+', [1]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('+', 1)]


def test_deflate_single_elements3():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(2, [ ('0', [1]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('0', 1)]


def test_deflate_single_elements4():
    """deflate indices does nothing for single elements"""
    fop = FermionicOperator(2, [ ('1', [0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('1', 0)]


def test_deflate_pairs1():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('--', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs2():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('-+', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('0', 0)]


def test_deflate_pairs3():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('-0', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs4():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('-1', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('-', 0)]


def test_deflate_pairs5():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('+-', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('1', 0)]


def test_deflate_pairs6():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('++', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs7():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('+0', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('+', 0)]


def test_deflate_pairs8():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('+1', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs9():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('0-', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('-', 0)]


def test_deflate_pairs10():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('0+', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs11():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('00', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('0', 0)]


def test_deflate_pairs12():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('01', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs13():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('1-', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None


def test_deflate_pairs14():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('1+', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == [('+', 0)]


def test_deflate_pairs15():
    """deflate indices pair testing"""
    fop = FermionicOperator(1, [ ('10', [0,0]) ])
    fop_deflate = fop.deflate_indices()
    assert fop_deflate.operators == None
