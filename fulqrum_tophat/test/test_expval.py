# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test expectation values"""

import fulqrum_tophat as top

PROBS = {'1000': 0.0022,
         '1001': 0.0045,
         '1110': 0.0081,
         '0001': 0.0036,
         '0010': 0.0319,
         '0101': 0.001,
         '1100': 0.0008,
         '1010': 0.0009,
         '1111': 0.3951,
         '0011': 0.0007,
         '0111': 0.01,
         '0000': 0.4666,
         '1101': 0.0355,
         '1011': 0.0211,
         '0110': 0.0081,
         '0100': 0.0099
         }

def test_basic_expvals1():
    """Test that basic exp values work 1"""
    op = top.QubitOperator(2, (('Z', 0), ('Z', 1)))
    dist = {'00': 0.5, '11': 0.5}
    assert op.expval(dist) == 1.0


def test_basic_expvals2():
    """Test that basic exp values work 2"""
    op = top.QubitOperator(3, (('Z', 0), ('Z', 1), ('Z', 2)))
    dist = {'000': 0.5, '111': 0.5}
    assert op.expval(dist) == 0


def test_basic_expvals3():
    """Test that basic exp values work 3"""
    op = top.QubitOperator(3, ())
    dist = {'000': 0.5, '111': 0.5}
    assert op.expval(dist) == 1


def test_basic_expvals4():
    """Test that basic exp values work 4"""
    op = top.QubitOperator(2, [('Z', 1)])
    dist = {'000': 0.5, '111': 0.5}
    assert op.expval(dist) == 0


def test_basic_expvals5():
    """Test that basic exp values work 5"""
    op = top.QubitOperator(2, [('Z', 0)])
    dist = {'000': 0.5, '111': 0.5}
    assert op.expval(dist) == 0


def test_generic_zzzz_probs():
    """Test ZZZZ with PROBS"""
    op = top.QubitOperator(4, (('Z', kk) for kk in range(4)))
    assert abs(op.expval(PROBS) - 0.7554) < 1e-15


def test_asymmetric_operators():
    """Test asymmetric operators give correct values"""
    op = top.QubitOperator(4, [('0', 3)])
    assert abs(op.expval(PROBS) - 0.5318) < 1e-15

    op2 = top.QubitOperator(4, [('0', 0)])
    assert abs(op2.expval(PROBS) - 0.5285) < 1e-15

    op3 = top.QubitOperator(4, [('1',0), ('1', 1), ('0', 2), ('1', 3)])
    assert abs(op3.expval(PROBS) - 0.0211) < 1e-15
