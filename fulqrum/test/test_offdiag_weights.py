# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test off-diagonal weights"""
import numpy as np
import fulqrum as fq
from fulqrum import QubitOperator


def test_offdiag_weights1():
    Q = QubitOperator.from_label("IIIII")
    assert np.allclose(Q.offdiag_weights(), np.array([0]))


def test_offdiag_weights2():
    Q = QubitOperator.from_label("IIZII")
    assert np.allclose(Q.offdiag_weights(), np.array([0]))


def test_offdiag_weights3():
    Q = QubitOperator.from_label("IIYII")
    assert np.allclose(Q.offdiag_weights(), np.array([1]))


def test_offdiag_weights4():
    Q = QubitOperator.from_label("+ZYZX")
    assert np.allclose(Q.offdiag_weights(), np.array([3]))


def test_offdiag_weights5():
    Q = QubitOperator.from_label("0IYI1") + QubitOperator.from_label("+IXI-")
    assert np.allclose(Q.offdiag_weights(), np.array([1, 3]))


def test_offdiag_weights6():
    Q = QubitOperator.from_label("IIYII") + QubitOperator.from_label("IIXI-")
    assert np.allclose(Q.offdiag_weights(), np.array([1, 2]))


def test_offdiag_weights7():
    N = 5
    Q = QubitOperator(N, [("Y", [2], 1)]) + QubitOperator(N, [("-X", [0, 2], 5)])
    assert np.allclose(Q.offdiag_weights(), np.array([1, 2]))


def test_offdiag_weights8():
    N = 5
    Q = QubitOperator(N, [("Y", [2], 1)]) + QubitOperator(
        N, [("-XY", [4, 0, 2], -1 + 1j)]
    )
    assert np.allclose(Q.offdiag_weights(), np.array([1, 3]))


def test_offdiag_weight_sorting():
    """Test off-diagonal weight sorting"""
    op = QubitOperator.from_label("IXI")
    op += QubitOperator.from_label("YXX")
    op += QubitOperator.from_label("X1Y")
    op += QubitOperator.from_label("ZI0")
    op.offdiag_weight_sort()
    assert np.allclose(op.offdiag_weights(), [0, 1, 2, 3])


def test_offdiag_weight_ptrs1():
    """Test pointers when all offdiag weights the same"""
    op = fq.QubitOperator.from_label("IIII+")
    op += fq.QubitOperator.from_label("III+I")
    op += fq.QubitOperator.from_label("II+II")
    op += fq.QubitOperator.from_label("I+III")
    op += fq.QubitOperator.from_label("+IIII")
    assert np.allclose(op.offdiag_weight_ptrs(), [0, 5])


def test_offdiag_weight_ptrs3():
    """Test pointers for mix of diag and off-diag"""
    op = fq.QubitOperator.from_label("IIIII")
    op += fq.QubitOperator.from_label("IIZII")
    op += fq.QubitOperator.from_label("IZZZI")
    op += fq.QubitOperator.from_label("I+III")
    op += fq.QubitOperator.from_label("++III")
    assert np.allclose(op.offdiag_weight_ptrs(), [3, 4, 5])
    assert np.allclose(op[:3].offdiag_weights(), [0, 0, 0])
    assert np.allclose(op[3].offdiag_weights(), [1])
    assert np.allclose(op[4].offdiag_weights(), [2])


def test_offdiag_weight_ptrs_all_diag():
    """Test pointers when all terms are diagonal"""
    op = fq.QubitOperator.from_label("IIIII")
    op += fq.QubitOperator.from_label("IIZII")
    op += fq.QubitOperator.from_label("IZZZI")
    assert np.allclose(op.offdiag_weight_ptrs(), [])
