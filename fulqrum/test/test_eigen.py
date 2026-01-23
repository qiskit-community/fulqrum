# This code is a Qiskit project.
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
"""Test matvec functionality"""

import numpy as np
from fulqrum import QubitOperator, Subspace, SubspaceHamiltonian
from fulqrum.utils import kron_str

import scipy.sparse.linalg as spla


def grab_subspace(A, rows):
    B = np.zeros((len(rows), len(rows)), dtype=complex)
    for idx, ii in enumerate(rows):
        for jdx, jj in enumerate(rows):
            B[idx, jdx] = A[ii, jj]
    return B


def test_eigen1():
    """Test simple eigen"""
    obs = ["ZXIZ", "YIIZ", "XI-X", "XI+X"]
    weights = [1, -2, 7, 7]
    rows = [0, 3, 4, 7, 11, 12, 14, 15]

    width = len(obs[0])
    H = QubitOperator(width)
    for op, weight in zip(obs, weights):
        H += weight * QubitOperator.from_label(op)

    A = 0
    for op, weight in zip(obs, weights):
        A += weight * kron_str(op)
    assert np.allclose(A, A.conj().T)

    B = grab_subspace(A, rows)

    subspace_dict = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    ans_evals, _ = spla.eigsh(B, k=2, which="SA")

    S = Subspace([list(subspace_dict.keys())])
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=2, which="SA")
    assert np.allclose(ans_evals, evals)
    for kk in range(2):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )

    # hashing only the first (`bitset.m_bits[0]`) bitset block
    S = Subspace([list(subspace_dict.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=2, which="SA")
    assert np.allclose(ans_evals, evals)
    for kk in range(2):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )


def test_eigen2():
    """Test simple diag only"""
    obs = ["ZIIIZ", "II0IZ", "11IZ0"]
    weights = [1, -2, -5]
    rows = [0, 3, 4, 12, 14, 15, 25, 31]

    width = len(obs[0])
    H = QubitOperator(width)
    for op, weight in zip(obs, weights):
        H += weight * QubitOperator.from_label(op)

    A = 0
    for op, weight in zip(obs, weights):
        A += weight * kron_str(op)
    assert np.allclose(A, A.conj().T)

    B = grab_subspace(A, rows)

    subspace_dict = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    ans_evals, _ = spla.eigsh(B, k=2, which="SA")

    S = Subspace([list(subspace_dict.keys())])
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=2, which="SA")
    assert np.allclose(ans_evals, evals)
    for kk in range(2):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )

    # hashing only the first (`bitset.m_bits[0]`) bitset block
    S = Subspace([list(subspace_dict.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=2, which="SA")
    assert np.allclose(ans_evals, evals)
    for kk in range(2):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )


def test_eigen3():
    """Test simple off-diag only"""
    obs = ["ZI+IZ", "ZI-IZ", "IYYIZ", "XXIZ0", "III-+", "III+-"]
    weights = [1, 1, -2, -5, 3.283j, -3.283j]
    rows = [0, 3, 4, 12, 14, 16, 17, 25, 31]

    width = len(obs[0])
    H = QubitOperator(width)
    for op, weight in zip(obs, weights):
        H += weight * QubitOperator.from_label(op)

    A = 0
    for op, weight in zip(obs, weights):
        A += weight * kron_str(op)
    assert np.allclose(A, A.conj().T)

    B = grab_subspace(A, rows)

    subspace_dict = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    ans_evals, _ = spla.eigsh(B, k=3, which="SA")

    S = Subspace([list(subspace_dict.keys())])
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=3, which="SA")
    assert np.allclose(ans_evals, evals)
    for kk in range(3):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )

    # hashing only the first (`bitset.m_bits[0]`) bitset block
    S = Subspace([list(subspace_dict.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=3, which="SA")
    assert np.allclose(ans_evals, evals)
    for kk in range(3):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )


def test_eigen4():
    """Test bunch of operators"""
    num_qubits = 5
    obs = ["ZI+IZ", "ZI-IZ", "IYYIZ", "XXIZ0", "III-+", "III+-"]
    weights = [5j + 4, -5j + 4, -2, -5, 3.283j, -3.283j]
    rows = [kk for kk in range(2**num_qubits - 5)]

    A = 0
    for op, weight in zip(obs, weights):
        A += weight * kron_str(op)
    assert np.allclose(A, A.conj().T)

    v0 = np.ones(len(rows), dtype=complex)
    B = grab_subspace(A, rows)
    ans_evals, ans_evecs = spla.eigsh(B, k=3, which="SA", v0=v0)

    width = len(obs[0])
    H = QubitOperator(width)
    for op, weight in zip(obs, weights):
        H += weight * QubitOperator.from_label(op)

    subspace_dict = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = Subspace([list(subspace_dict.keys())])
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=3, which="SA", v0=v0)
    assert np.allclose(ans_evals, evals)
    for kk in range(3):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )

    # hashing only the first (`bitset.m_bits[0]`) bitset block
    S = Subspace([list(subspace_dict.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=3, which="SA", v0=v0)
    assert np.allclose(ans_evals, evals)
    for kk in range(3):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )


def test_eigen5():
    """Diagonal terms only"""
    num_qubits = 5
    num_evals = 3
    obs = ["01011", "IIIII", "ZZIZZ"]
    weights = [(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(obs))]

    rows = [kk for kk in range(10, 2**num_qubits, 2)]

    A = 0
    for op, weight in zip(obs, weights):
        A += weight * kron_str(op)
    assert np.allclose(A, A.conj().T)

    v0 = np.ones(len(rows), dtype=float)
    B = grab_subspace(A, rows)
    ans_evals, ans_evecs = spla.eigsh(B, k=num_evals, which="SA", v0=v0)

    width = len(obs[0])
    H = QubitOperator(width)
    for op, weight in zip(obs, weights):
        H += weight * QubitOperator.from_label(op)

    subspace_dict = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = Subspace([list(subspace_dict.keys())])
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=num_evals, which="SA", v0=v0)
    assert np.allclose(ans_evals, evals)
    for kk in range(num_evals):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )

    # hashing only the first (`bitset.m_bits[0]`) bitset block
    S = Subspace([list(subspace_dict.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=num_evals, which="SA", v0=v0)
    assert np.allclose(ans_evals, evals)
    for kk in range(num_evals):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )


def test_eigen6():
    """Random stuff"""
    num_qubits = 5
    num_evals = 3
    obs = [
        "XIXII",
        "+1IZI",
        "-1IZI",
        "0101I",
        "II0II",
        "XYIYX",
        "III-+",
        "III+-",
        "ZZZZZ",
    ]
    weights = [1, 1 - 2j, 1 + 2j, -3, 1.25, -1, 2, 2, 4]

    rows = [kk for kk in range(2, 2**num_qubits, 3)]

    A = 0
    for op, weight in zip(obs, weights):
        A += weight * kron_str(op)
    assert np.allclose(A, A.conj().T)

    v0 = np.ones(len(rows), dtype=complex)
    B = grab_subspace(A, rows)
    ans_evals, ans_evecs = spla.eigsh(B, k=num_evals, which="SA", v0=v0)

    width = len(obs[0])
    H = QubitOperator(width)
    for op, weight in zip(obs, weights):
        H += weight * QubitOperator.from_label(op)

    subspace_dict = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = Subspace([list(subspace_dict.keys())])
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=num_evals, which="SA", v0=v0)
    assert np.allclose(ans_evals, evals)
    for kk in range(num_evals):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )

    # hashing only the first (`bitset.m_bits[0]`) bitset block
    S = Subspace([list(subspace_dict.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=num_evals, which="SA", v0=v0)
    assert np.allclose(ans_evals, evals)
    for kk in range(num_evals):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )


def test_diagonal_type2():
    """Test that a diagonal type=2 Hamiltionian can generate CSR"""
    op = QubitOperator.from_label("IZIZZ")
    op.set_type(2)

    S = Subspace([["00000", "11111"]])

    Hsub = SubspaceHamiltonian(op, S)

    M = Hsub.to_csr_linearoperator_fast().matrix
    assert M.nnz == 2
    assert np.allclose(M.data, np.array([1, -1], dtype=float))
