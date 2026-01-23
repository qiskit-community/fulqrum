# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=no-name-in-module
"""Test basic core functionality"""
import numpy as np
import scipy.sparse as sp
from fulqrum import QubitOperator

X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

I = np.array([[1, 0], [0, 1]], dtype=complex)

P = np.array([[0, 0], [1, 0]], dtype=complex)

M = np.array([[0, 1], [0, 0]], dtype=complex)

G = np.array([[1, 0], [0, 0]], dtype=complex)

O = np.array([[0, 0], [0, 1]], dtype=complex)


def _build_dense_matrix(op):
    """Build dense matrix from matrix_element method using ints"""
    N = op.width
    A = np.zeros((2**N, 2**N), dtype=complex)
    for row_idx in range(2**N):
        for col_idx in range(2**N):
            A[row_idx, col_idx] = op.matrix_element(row_idx, col_idx)
    return A


def test_build_full1():
    """Validate that correct matrix is returned"""
    op = QubitOperator.from_label("X" * 3)
    A = _build_dense_matrix(op)
    B = sp.kron(X, sp.kron(X, X)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full2():
    """Validate that correct matrix is returned"""
    op = QubitOperator.from_label("IXZ")
    A = _build_dense_matrix(op)
    B = sp.kron(I, sp.kron(X, Z)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full3():
    """Validate that correct matrix is returned"""
    op = QubitOperator.from_label("IXY")
    A = _build_dense_matrix(op)
    B = sp.kron(I, sp.kron(X, Y)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full4():
    """Validate that correct matrix is returned"""
    op = QubitOperator.from_label("+0Y")
    A = _build_dense_matrix(op)
    B = sp.kron(P, sp.kron(G, Y)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full5():
    """Validate that correct matrix is returned"""
    op = QubitOperator.from_label("III")
    A = _build_dense_matrix(op)
    B = sp.kron(I, sp.kron(I, I)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full_string():
    """Validate can build from string input"""
    op = QubitOperator.from_label("+0Y")
    N = op.width
    counts = {}
    for kk in range(2**N):
        counts[bin(kk)[2:].zfill(N)] = kk

    A = np.zeros((2**N, 2**N), dtype=complex)
    for row_idx, row_str in enumerate(counts):
        for col_idx, col_str in enumerate(counts):
            A[row_idx, col_idx] = op.matrix_element(row_str, col_str)
    B = sp.kron(P, sp.kron(G, Y)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full_multi_term1():
    """Validate that correct matrix is returned from multi terms"""
    op = QubitOperator.from_label("III")
    op += QubitOperator.from_label("III")
    A = _build_dense_matrix(op)
    B = 2 * sp.kron(I, sp.kron(I, I)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full_multi_term2():
    """Validate that correct matrix is returned from multi terms"""
    op = QubitOperator.from_label("III", -1j)
    op += QubitOperator.from_label("III")
    A = _build_dense_matrix(op)
    B = (1 - 1j) * sp.kron(I, sp.kron(I, I)).toarray()
    assert np.linalg.norm(A - B) < 1e-15


def test_build_full_multi_term3():
    """Validate that correct matrix is returned from multi terms"""
    op = QubitOperator.from_label("III", -1j)
    op += QubitOperator.from_label("XY-", 5)
    A = _build_dense_matrix(op)
    B = (
        -1j * sp.kron(I, sp.kron(I, I)).toarray()
        + 5 * sp.kron(X, sp.kron(Y, M)).toarray()
    )
    assert np.linalg.norm(A - B) < 1e-15
