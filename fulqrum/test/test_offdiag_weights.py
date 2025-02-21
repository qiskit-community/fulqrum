# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test off-diagonal weights"""
import numpy as np
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
