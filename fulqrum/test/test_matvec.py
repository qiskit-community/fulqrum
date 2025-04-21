# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test matvec functionality"""

import numpy as np
from fulqrum import QubitOperator, Subspace, SubspaceHamiltonian
from fulqrum.utils import kron_str


def test_matvec1():
    """Test simple matvec over full subspace"""
    H = QubitOperator.from_label("ZZ")
    H += QubitOperator.from_label("XX", 5)
    H += QubitOperator.from_label("YY", -3)
    S = Subspace({"00": 10, "01": 10, "10": 10, "11": 10})
    F = SubspaceHamiltonian(H, S)
    in_vec = np.ones(len(S), dtype=complex)
    perm = [int(key, 2) for key in S.to_dict().keys()]
    perm_vec = in_vec[perm]
    out_vec = F.matvec(perm_vec)
    res = F.interpret_vector(out_vec, -1, sort=True)
    dense_op = kron_str("ZZ") + 5 * kron_str("XX") - 3 * kron_str("YY")
    ans = dense_op.dot(in_vec)
    assert np.allclose(list(res.values()), ans)


def test_matvec2():
    """Test simple matvec over full subspace for ID ops"""
    H = QubitOperator.from_label("II")
    S = Subspace({"00": 10, "01": 10, "10": 10, "11": 10})
    F = SubspaceHamiltonian(H, S)
    in_vec = np.arange(len(S), dtype=complex)
    perm = [int(key, 2) for key in S.to_dict().keys()]
    perm_vec = in_vec[perm]
    out_vec = F.matvec(perm_vec)
    res = F.interpret_vector(out_vec, -1, sort=True)
    dense_op = kron_str("II")
    ans = dense_op.dot(in_vec)
    assert np.allclose(list(res.values()), ans)


def test_matvec3():
    """Test simple matvec over full subspace for ID ops"""
    H = QubitOperator.from_label("YX")
    H += 4 / 5 * QubitOperator.from_label("+I")
    S = Subspace({"00": 10, "01": 10, "10": 10, "11": 10})
    F = SubspaceHamiltonian(H, S)
    in_vec = np.arange(len(S), dtype=complex)
    perm = [int(key, 2) for key in S.to_dict().keys()]
    perm_vec = in_vec[perm]
    out_vec = F.matvec(perm_vec)
    res = F.interpret_vector(out_vec, -1, sort=True)
    dense_op = kron_str("YX") + 4 / 5 * kron_str("+I")
    ans = dense_op.dot(in_vec)
    assert np.allclose(list(res.values()), ans)


def test_matvec4():
    """Test simple matvec over truncated subspace"""
    H = QubitOperator.from_label("YX")
    H += 4 / 5 * QubitOperator.from_label("+I")
    S = Subspace({"00": 10, "10": 10})
    F = SubspaceHamiltonian(H, S)
    in_vec = np.ones(len(S), dtype=complex)
    out_vec = F.matvec(in_vec)
    res = F.interpret_vector(out_vec, -1, sort=True)
    dense_op = kron_str("YX") + 4 / 5 * kron_str("+I")
    ans = dense_op[0:3:2, 0:3:2].dot(in_vec)
    assert np.allclose(list(res.values()), ans)


def test_matvec5():
    """Test simple matvec over truncated subspace"""
    H = QubitOperator.from_label("YX")
    H += 4 / 5 * QubitOperator.from_label("+I")
    S = Subspace({"00": 10, "11": 10})
    F = SubspaceHamiltonian(H, S)
    in_vec = np.ones(len(S), dtype=complex)
    out_vec = F.matvec(in_vec)
    res = F.interpret_vector(out_vec, -1, sort=True)
    dense_op = kron_str("YX") + 4 / 5 * kron_str("+I")
    ans = dense_op[0:4:3, 0:4:3].dot(in_vec)
    assert np.allclose(list(res.values()), ans)


def test_matvec_bin_width1():
    """Validate that matvec is unchanged with bin_width for diagonal op"""
    counts = {}
    ans_dict = {}
    in_vec = np.ones(2**3, dtype=complex)
    diag = kron_str("ZIZ").dot(in_vec)
    idx = 0
    for kk in range(2**3):
        counts[bin(kk)[2:].zfill(3)] = 1
        ans_dict[bin(kk)[2:].zfill(3)] = diag[idx]
        idx += 1

    H = QubitOperator.from_label("ZIZ")
    for bin_width in range(1, 4):
        S = Subspace(counts, bin_width=bin_width)
        sub = SubspaceHamiltonian(H, S)
        res_dict = sub.interpret_vector(sub.matvec(in_vec), -1)
        assert res_dict == ans_dict


def test_matvec_bin_width2():
    """Validate that matvec is unchanged with bin_width for diagonal + off_diag op"""
    counts = {}
    ans_dict = {}
    in_vec = np.ones(2**3, dtype=complex)
    diag = (kron_str("ZIZ") + kron_str("XXX")).dot(in_vec)
    idx = 0
    for kk in range(2**3):
        counts[bin(kk)[2:].zfill(3)] = 1
        ans_dict[bin(kk)[2:].zfill(3)] = diag[idx]
        idx += 1

    H = QubitOperator.from_label("ZIZ") + QubitOperator.from_label("XXX")
    for bin_width in range(1, 4):
        S = Subspace(counts, bin_width=bin_width)
        sub = SubspaceHamiltonian(H, S)
        res_dict = sub.interpret_vector(sub.matvec(in_vec), -1)
        assert res_dict == ans_dict


def test_matvec_bin_width3():
    """Validate that matvec is unchanged with bin_width for diagonal + off_diag op and coeff"""
    counts = {}
    ans_dict = {}
    in_vec = np.ones(2**3, dtype=complex)
    diag = (kron_str("ZIZ") + -1j * kron_str("XXX")).dot(in_vec)
    idx = 0
    for kk in range(2**3):
        counts[bin(kk)[2:].zfill(3)] = 1
        ans_dict[bin(kk)[2:].zfill(3)] = diag[idx]
        idx += 1

    H = QubitOperator.from_label("ZIZ") + -1j * QubitOperator.from_label("XXX")
    for bin_width in range(1, 4):
        S = Subspace(counts, bin_width=bin_width)
        sub = SubspaceHamiltonian(H, S)
        res_dict = sub.interpret_vector(sub.matvec(in_vec), -1)
        assert res_dict == ans_dict


def test_matvec_bin_width4():
    """Validate that matvec is unchanged with bin_width and more interesting ops"""
    counts = {}
    ans_dict = {}
    diag = (kron_str("0IZ") + kron_str("XY+")).dot(np.ones(2**3, dtype=complex))
    idx = 0
    for kk in range(2**3):
        counts[bin(kk)[2:].zfill(3)] = 1
        ans_dict[bin(kk)[2:].zfill(3)] = diag[idx]
        idx += 1

    H = QubitOperator.from_label("0IZ") + QubitOperator.from_label("XY+")
    for bin_width in range(1, 4):
        S = Subspace(counts, bin_width=bin_width)
        sub = SubspaceHamiltonian(H, S)
        res_dict = sub.interpret_vector(sub.matvec(np.ones(2**3, dtype=complex)), -1)
        assert res_dict == ans_dict
