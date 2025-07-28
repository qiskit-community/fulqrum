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


_path = Path(__file__).parent / "data/h2.json"
FOP = FermionicOperator.from_json(_path)
OP = FOP.extended_jw_transformation()
M = qubitoperator_to_matrix(OP)
ANS_EVALS, ANS_EVECS = la.eigh(M)
ANS_EVECS = np.real(ANS_EVECS)
GROUND_ENERGY = ANS_EVALS[0]

GROUND_DIST = {}
for state_int in np.where(ANS_EVECS[:, 0] != 0)[0]:
    GROUND_DIST[bin(state_int)[2:].zfill(4)] = ANS_EVECS[state_int, 0]


def test_full_dist_h2_eigen():
    """Test full space solution against exact"""
    full_dist = {}
    for kk in range(2**4):
        full_dist[bin(kk)[2:].zfill(4)] = None

    S = Subspace(full_dist)
    Hsub = SubspaceHamiltonian(OP, S)

    # here we use starting vector of all ones to match phase with direct ans
    x0 = np.ones(len(S), dtype=float if OP.is_real() else complex)
    evals, evecs = spla.eigsh(Hsub, k=1, which="SA", v0=x0)
    assert np.allclose(evals, GROUND_ENERGY)
    assert np.allclose(evecs.ravel(), ANS_EVECS[:, 0])


def test_partial_dist_h2_eigen1():
    """Test subspace that overlaps with ground state still works"""
    part_dist = GROUND_DIST.copy()
    part_dist["0000"] = 1

    S = Subspace(part_dist)
    Hsub = SubspaceHamiltonian(OP, S)

    # here we use starting vector of all ones to match phase with direct ans
    x0 = np.ones(len(S), dtype=float if OP.is_real() else complex)
    evals, evecs = spla.eigsh(Hsub, k=1, which="SA", v0=x0)
    assert np.allclose(evals, GROUND_ENERGY)
    ans_dict = Hsub.interpret_vector(evecs)
    assert abs(ans_dict["1010"] - GROUND_DIST["1010"]) < 1e-14
    assert abs(ans_dict["0101"] - GROUND_DIST["0101"]) < 1e-14


def test_partial_dist_h2_eigen2():
    """Test subspace that overlaps with ground state still works"""
    part_dist = GROUND_DIST.copy()
    part_dist["1111"] = 1

    S = Subspace(part_dist)
    Hsub = SubspaceHamiltonian(OP, S)

    # here we use starting vector of all ones to match phase with direct ans
    x0 = np.ones(len(S), dtype=float if OP.is_real() else complex)
    evals, evecs = spla.eigsh(Hsub, k=1, which="SA", v0=x0)
    assert np.allclose(evals, GROUND_ENERGY)
    ans_dict = Hsub.interpret_vector(evecs)
    assert abs(ans_dict["1010"] - GROUND_DIST["1010"]) < 1e-14
    assert abs(ans_dict["0101"] - GROUND_DIST["0101"]) < 1e-14


def test_full_dist_h2_eigen_csr_linearoperator():
    """Test full space solution against exact for CSR linearoperator"""
    full_dist = {}
    for kk in range(2**4):
        full_dist[bin(kk)[2:].zfill(4)] = None

    S = Subspace(full_dist)
    Hsub = SubspaceHamiltonian(OP, S)
    M = Hsub.to_csr_linearoperator()

    # here we use starting vector of all ones to match phase with direct ans
    x0 = np.ones(len(S), dtype=float if OP.is_real() else complex)
    evals, evecs = spla.eigsh(M, k=1, which="SA", v0=x0)
    assert np.allclose(evals, GROUND_ENERGY)
    assert np.allclose(evecs.ravel(), ANS_EVECS[:, 0])
