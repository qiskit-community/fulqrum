# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test basic core functionality"""

from fulqrum_tophat import QubitOperator


def test_empty_qubitoperator():
    """Test empty QubitOperator"""
    N = 5
    qo = QubitOperator(N)
    qo.num_terms
    assert qo.num_terms == 0


def test_identity():
    """Test identity QubitOperator"""
    N = 5
    qo = QubitOperator(N, ())
    assert qo.num_terms == 1


def test_identity2():
    """Test identity QubitOperator 2"""
    N = 5
    qo = QubitOperator(N, [])
    assert qo.num_terms == 1


def test_coeff():
    """Test setting coeff for single term operator"""
    N = 5
    qo = QubitOperator(N, [], coeff=1 + 2j)
    assert qo.coeff == 1 + 2j


def test_simple_multi_operators():
    """Test simple operator with non-identity weight"""
    N = 5
    qo = QubitOperator(N, (("X", kk) for kk in range(N)))
    assert qo.operators == [("X", kk) for kk in range(N)]


def test_simple_multi_weight():
    """Test simple operator with non-identity weight"""
    N = 5
    qo = QubitOperator(N, (("X", kk) for kk in range(N)))
    assert qo.weight() == N


def test_simple_inplace_add():
    """Test that inplace addition gives expected results"""
    N = 5
    op = QubitOperator(N)

    for kk in range(N):
        op += QubitOperator(N, [("Y", kk)], coeff=1.0 / (N + kk))

    for kk in range(N):
        assert op[kk].operators == [("Y", kk)]
        assert op[kk].coeff == 1.0 / (N + kk)


def test_simple_add():
    """Test that addition gives expected results"""
    N = 5
    op = QubitOperator(N)

    for kk in range(N):
        op = op + QubitOperator(N, [("0", kk)], coeff=1.0 / (N + kk))

    for kk in range(N):
        assert op[kk].operators == [("0", kk)]
        assert op[kk].coeff == 1.0 / (N + kk)


def test_simple_diagonal():
    """Verify diagonal operator returns True"""
    N = 25
    op = QubitOperator(N)
    diag_ops = ["I", "Z", "0", "1"]

    for kk in range(20):
        op += QubitOperator(N, [(diag_ops[kk % len(diag_ops)], kk)], coeff=1 / (N + kk))

    assert op.is_diagonal() is True


def test_simple_diagonal2():
    """Verify non-diagonal operator returns False"""
    N = 25
    op = QubitOperator(N)
    diag_ops = ["I", "Z", "0", "1"]

    for kk in range(20):
        if kk:
            op += QubitOperator(
                N, [(diag_ops[kk % len(diag_ops)], kk)], coeff=1 / (N + kk)
            )
        # Add non-diagonal X operator when kk = 0
        else:
            op += QubitOperator(N, [("X", kk)], coeff=1 / (N + kk))

    assert op.is_diagonal() is False


def test_operator_diagonal_splitting():
    """Test operator diagonal splitting"""
    diag_ops = ["I", "X", "Z", "0", "Y", "1"]
    N = len(diag_ops)
    op = QubitOperator(N)

    for kk, oper in enumerate(diag_ops):
        op += QubitOperator(N, [(oper, 0)], coeff=1 / (N + kk))
    diag, off_diag = op.split_diagonal()

    assert diag.num_terms == 4
    assert off_diag.num_terms == 2


def test_operator_identity_removal():
    """Test operator identity values and removal"""
    diag_ops = ["I", "X", "Z", "0", "Y", "1"]
    N = len(diag_ops)
    op = QubitOperator(N)

    for kk, oper in enumerate(diag_ops):
        op += QubitOperator(N, [(oper, 0)], coeff=1 / (N + kk))

    assert op.identity_terms_value() == 0.16666666666666666
    new_op = op.remove_identity_terms()
    assert new_op.num_terms == 5
    assert new_op.identity_terms_value() == 0
