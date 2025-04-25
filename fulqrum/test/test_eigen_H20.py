# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test eigen functionality on H2"""
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from fulqrum import FermionicOperator, Subspace, SubspaceHamiltonian
from fulqrum.utils import qubitoperator_to_matrix


_path = Path(__file__).parent / "data/h2o.json"
FOP = FermionicOperator.from_json(_path)
OP = FOP.extended_jw_transformation()
NEW_OP = OP.combine_repeated_terms()
GROUND_ENERGY = -84.20635059 # Answer from direct full-matrix 


def test_full_dist_h20_eigenenergy_matrix_free():
    """Test full space solution against exact"""
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    S = Subspace(full_dist)
    Hsub = SubspaceHamiltonian(NEW_OP, S)

    evals, _ = spla.eigsh(Hsub, k=1, which="SA", v0=np.ones(len(S), dtype=complex))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_full_dist_h20_eigenenergy_csr():
    """Test full space solution against exact"""
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    S = Subspace(full_dist)
    Hsub = SubspaceHamiltonian(NEW_OP, S)
    M = Hsub.to_csr_array()

    evals, _ = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=complex))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)
