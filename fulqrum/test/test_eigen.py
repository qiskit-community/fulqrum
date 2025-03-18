# Fulqrum
# Copyright (C) 2024, IBM
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

    for bin_width in range(width):
        S = Subspace(subspace_dict, bin_width=bin_width)
        Hsub = SubspaceHamiltonian(H, S)
        evals, _ = spla.eigsh(Hsub, k=2, which="SA")
        assert np.allclose(ans_evals, evals)

    # validate eigenvectors with full binning
    S = Subspace(subspace_dict, width)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=2, which="SA")
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

    for bin_width in range(width):
        S = Subspace(subspace_dict, bin_width=bin_width)
        Hsub = SubspaceHamiltonian(H, S)
        evals, _ = spla.eigsh(Hsub, k=2, which="SA")
        assert np.allclose(ans_evals, evals)

    # validate eigenvectors with full binning
    S = Subspace(subspace_dict, width)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=2, which="SA")
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

    for bin_width in range(width):
        S = Subspace(subspace_dict, bin_width=bin_width)
        Hsub = SubspaceHamiltonian(H, S)
        evals, _ = spla.eigsh(Hsub, k=3, which="SA")
        assert np.allclose(ans_evals, evals)

    # validate eigenvectors with full binning
    S = Subspace(subspace_dict, width)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=3, which="SA")
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
    for bin_width in range(width):
        S = Subspace(subspace_dict)
        Hsub = SubspaceHamiltonian(H, S)
        evals, evecs = spla.eigsh(Hsub, k=3, which="SA", v0=v0)
        assert np.allclose(ans_evals, evals)

    # validate eigenvectors with full binning
    S = Subspace(subspace_dict, width)
    Hsub = SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=3, which="SA", v0=v0)
    for kk in range(3):
        assert (
            np.linalg.norm(B.dot(evecs[:, kk]) - evals[kk] * evecs[:, kk], np.inf)
            < 1e-13
        )
