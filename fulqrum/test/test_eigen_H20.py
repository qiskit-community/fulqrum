# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test eigen functionality on H2"""
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from fulqrum import FermionicOperator, Subspace, SubspaceHamiltonian


_path = Path(__file__).parent / "data/h2o.json"
FOP = FermionicOperator.from_json(_path)
NEW_OP = FOP.extended_jw_transformation()
GROUND_ENERGY = -84.20635059  # Answer from direct full-matrix


def test_full_dist_h20_eigenenergy_matrix_free():
    """Test full space solution against exact"""
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    S = Subspace([list(full_dist.keys())])
    Hsub = SubspaceHamiltonian(NEW_OP, S)

    evals, _ = spla.eigsh(Hsub, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)

    # single bitset block
    S = Subspace([list(full_dist.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(NEW_OP, S)

    evals, _ = spla.eigsh(Hsub, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_full_dist_h20_eigenenergy_csr():
    """Test full space solution against exact for CSR matrix"""
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    S = Subspace([list(full_dist.keys())])
    Hsub = SubspaceHamiltonian(NEW_OP, S)
    M = Hsub.to_csr_array()
    assert M.dtype == float

    evals, _ = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)

    # single bitset block
    S = Subspace([list(full_dist.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(NEW_OP, S)
    M = Hsub.to_csr_array()
    assert M.dtype == float

    evals, _ = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_full_dist_h20_eigenenergy_csr_linearoperator():
    """Test full space solution against exact for CSR linearoperator"""
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    S = Subspace([list(full_dist.keys())])
    Hsub = SubspaceHamiltonian(NEW_OP, S)
    M = Hsub.to_csr_linearoperator()

    assert M.matrix.dtype == float
    evals, _ = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)

    # single bitset block
    S = Subspace([list(full_dist.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(NEW_OP, S)
    M = Hsub.to_csr_linearoperator()

    assert M.matrix.dtype == float
    evals, _ = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_full_dist_h20_eigenenergy_csr_linearoperator_fast():
    """Test full space solution against exact for CSR linearoperator fast"""
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    S = Subspace([list(full_dist.keys())])
    Hsub = SubspaceHamiltonian(NEW_OP, S)
    M = Hsub.to_csr_linearoperator_fast()

    assert M.matrix.dtype == float
    evals, _ = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)

    # single bitset block
    S = Subspace([list(full_dist.keys())], use_all_bitset_blocks=False)
    Hsub = SubspaceHamiltonian(NEW_OP, S)
    M = Hsub.to_csr_linearoperator_fast()

    assert M.matrix.dtype == float
    evals, _ = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=float))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_proj_indices_set():
    """Test that projector indices set properly after JW transform"""
    for kk in range(NEW_OP.num_terms):
        has_proj_ops = 0
        for op_idx_pair in NEW_OP[kk].operators:
            if op_idx_pair[0] in ["0", "1"]:
                has_proj_ops = 1
                break
        if has_proj_ops:
            assert NEW_OP[kk].proj_indices.shape[0] > 0
        else:
            assert NEW_OP[kk].proj_indices.shape[0] == 0
