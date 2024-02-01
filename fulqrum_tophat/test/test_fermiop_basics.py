# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test basic core functionality"""

from fulqrum_tophat import FermionicOperator


def test_empty_fermioperator():
    """Test empty FermionicOperator"""
    N = 5
    fo = FermionicOperator(N)
    fo.num_terms
    assert fo.num_terms == 0


def test_identity():
    """Test identity FermionicOperator"""
    N = 5
    fo = FermionicOperator(N, ())
    assert fo.num_terms == 1


def test_identity2():
    """Test identity FermionicOperator 2"""
    N = 5
    fo = FermionicOperator(N, [])
    assert fo.num_terms == 1


def test_coeff():
    """Test setting coeff for single term operator"""
    N = 5
    fo = FermionicOperator(N, [], coeff=1 + 2j)
    assert fo.coeff == 1 + 2j


def test_simple_multi_operators():
    """Test simple operator with non-identity weight"""
    N = 5
    fo = FermionicOperator(N, (("+", kk) for kk in range(N)))
    assert fo.operators == [("+", kk) for kk in range(N)]


def test_simple_multi_weight():
    """Test simple operator with non-identity weight"""
    N = 5
    qo = FermionicOperator(N, (("-", kk) for kk in range(N)))
    assert qo.weight() == N


def test_simple_inplace_add():
    """Test that inplace addition gives expected results"""
    N = 5
    op = FermionicOperator(N)

    for kk in range(N):
        op += FermionicOperator(N, [("-", kk)], coeff=1.0 / (N + kk))

    for kk in range(N):
        assert op[kk].operators == [("-", kk)]
        assert op[kk].coeff == 1.0 / (N + kk)


def test_simple_add():
    """Test that addition gives expected results"""
    N = 5
    op = FermionicOperator(N)

    for kk in range(N):
        op = op + FermionicOperator(N, [("+", kk)], coeff=1.0 / (N + kk))

    for kk in range(N):
        assert op[kk].operators == [("+", kk)]
        assert op[kk].coeff == 1.0 / (N + kk)
