# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test matvec functionality"""

import numpy as np
import scipy.sparse as sp
import fulqrum as fq
from fulqrum.utils import kron_str


def grab_subspace(A, rows):
    B = np.zeros((len(rows), len(rows)), dtype=complex)
    for idx, ii in enumerate(rows):
        for jdx, jj in enumerate(rows):
            B[idx, jdx] = A[ii, jj]
    return B


def test_csr1():
    """Random terms CSR"""
    num_qubits = 5
    strings = ["XIXII", "+1IZI", "0101I", "II0II", "XYIYX", "III-+", "ZZZZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    M = sp.csr_array(A)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    # Full counts
    counts = {}
    for bits in range(2**num_qubits):
        counts[bin(bits)[2:].zfill(num_qubits)] = 1

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(2**num_qubits, dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr2():
    """Diagonal terms only CSR"""
    num_qubits = 5
    strings = ["01011", "IIIII", "ZZIZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    M = sp.csr_array(A)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    # Full counts
    counts = {}
    for bits in range(2**num_qubits):
        counts[bin(bits)[2:].zfill(num_qubits)] = 1

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(2**num_qubits, dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr3():
    """Off-diagonal, same structure CSR"""
    num_qubits = 5
    strings = ["XXIXX", "YYIYY", "-+I+-", "XYZXY"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    M = sp.csr_array(A)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    # Full counts
    counts = {}
    for bits in range(2**num_qubits):
        counts[bin(bits)[2:].zfill(num_qubits)] = 1

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(2**num_qubits, dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr4():
    """Off-diagonal, random structure CSR"""
    num_qubits = 5
    strings = ["XZIXX", "0Y+YZ", "-+X+-", "X0-1Y"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    M = sp.csr_array(A)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    # Full counts
    counts = {}
    for bits in range(2**num_qubits):
        counts[bin(bits)[2:].zfill(num_qubits)] = 1

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(2**num_qubits, dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr5():
    """Random stuff truncated CSR"""
    num_qubits = 5
    strings = ["XIXII", "+1IZI", "0101I", "II0II", "XYIYX", "III-+", "ZZZZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])
    rows = [0, 3, 4, 7, 11, 12, 14, 15, 27]

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    B = grab_subspace(A, rows)
    M = sp.csr_array(B)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(len(rows), dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr6():
    """Random stuff truncated CSR 2"""
    num_qubits = 5
    strings = ["XIXII", "+1IZI", "0101I", "II0II", "XYIYX", "III-+", "ZZZZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])
    rows = [0, 3, 11, 12, 14, 15, 19, 26, 27]

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    B = grab_subspace(A, rows)
    M = sp.csr_array(B)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(len(rows), dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr7():
    """Diagonals only truncated"""
    num_qubits = 5
    strings = ["01011", "IIIII", "ZZIZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])
    rows = [0, 3, 11, 12, 14, 15, 19, 26, 27]

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    B = grab_subspace(A, rows)
    M = sp.csr_array(B)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(len(rows), dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr8():
    """Test inner-loop exit condition CSR"""
    num_qubits = 5
    strings = ["YYIYY", "-+I+-"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])
    rows = [0, 1, 2, 3, 15, 19, 26, 27, 28]

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    B = grab_subspace(A, rows)
    M = sp.csr_array(B)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(len(rows), dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr9():
    """Off-diagonal, same structure truncated CSR"""
    num_qubits = 5
    strings = ["XZIXX", "0Y+YZ", "-+X+-", "X0-1Y"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])
    rows = [0, 1, 2, 3, 15, 19, 26, 27, 28]

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    B = grab_subspace(A, rows)
    M = sp.csr_array(B)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(len(rows), dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))


def test_csr10():
    """Zero elements in subspace CSR"""
    num_qubits = 5
    strings = ["XZIXX", "0Y+YZ", "-+X+-", "X0-1Y"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])
    rows = [0, 1, 2, 26, 27, 28]

    # build reference CSR
    A = 0
    for idx, string in enumerate(strings):
        A += values[idx] * kron_str(string)
    B = grab_subspace(A, rows)
    M = sp.csr_array(B)

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace(counts, num_qubits)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub.to_csr()

    assert np.allclose(P.data, M.data)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.indptr, M.indptr)

    x = np.ones(len(rows), dtype=complex)
    assert np.allclose(P.matvec(x), M.dot(x))
