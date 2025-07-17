# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test eigen functionality on H2"""
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

import fulqrum as fq


_path = Path(__file__).parent / "data/lih.json"
FOP = fq.FermionicOperator.from_json(_path)
OP = FOP.extended_jw_transformation()
GROUND_ENERGY = -7.875652564927877

GROUND_DIST = {
    "000011000011": (0.9910153728645456 + 0j),
    "000011000101": (0.029187890703293036 + 0j),
    "000011000110": (-0.0005491332647300028 + 0j),
    "000011100001": (-0.0004922075925036299 + 0j),
    "000011100010": (0.00041525316533507634 + 0j),
    "000011100100": (0.0026739676813965444 + 0j),
    "000101000011": (0.029187890703294108 + 0j),
    "000101000101": (-0.01982603883627099 + 0j),
    "000101000110": (0.00027650305574540096 + 0j),
    "000101100001": (0.04201885695185623 + 0j),
    "000101100010": (-0.0012816380340357537 + 0j),
    "000101100100": (-0.00018972324968284342 + 0j),
    "000110000011": (-0.0005491332647299678 + 0j),
    "000110000101": (0.0002765030557453956 + 0j),
    "000110000110": (-0.003543269120007031 + 0j),
    "000110100001": (-0.003955605715432327 + 0j),
    "000110100010": (-0.0014842073044949616 + 0j),
    "000110100100": (5.243203296635565e-05 + 0j),
    "001001001001": (-0.03569926754546147 + 0j),
    "001001001010": (0.002926468680405419 + 0j),
    "001001001100": (0.0002291575861529392 + 0j),
    "001001101000": (-0.00039959051518590236 + 0j),
    "001010001001": (0.002926468680405416 + 0j),
    "001010001010": (-0.0017410084706380115 + 0j),
    "001010001100": (-3.9858551395326815e-05 + 0j),
    "001010101000": (3.3923302022510734e-05 + 0j),
    "001100001001": (0.00022915758615294035 + 0j),
    "001100001010": (-3.985855139532961e-05 + 0j),
    "001100001100": (0.0001498794149102944 + 0j),
    "001100101000": (2.187359219366163e-05 + 0j),
    "010001010001": (-0.03569926754546147 + 0j),
    "010001010010": (0.0029264686804054175 + 0j),
    "010001010100": (0.00022915758615293336 + 0j),
    "010001110000": (-0.00039959051518590594 + 0j),
    "010010010001": (0.0029264686804054136 + 0j),
    "010010010010": (-0.001741008470638011 + 0j),
    "010010010100": (-3.985855139532696e-05 + 0j),
    "010010110000": (3.39233020225108e-05 + 0j),
    "010100010001": (0.00022915758615293588 + 0j),
    "010100010010": (-3.985855139532977e-05 + 0j),
    "010100010100": (0.0001498794149102968 + 0j),
    "010100110000": (2.1873592193661752e-05 + 0j),
    "011000011000": (0.00010970404096626051 + 0j),
    "100001000011": (-0.000492207592503681 + 0j),
    "100001000101": (0.042018856951856094 + 0j),
    "100001000110": (-0.003955605715432328 + 0j),
    "100001100001": (-0.09790795514329052 + 0j),
    "100001100010": (0.002784753449849705 + 0j),
    "100001100100": (0.0005586481007244836 + 0j),
    "100010000011": (0.00041525316533507926 + 0j),
    "100010000101": (-0.0012816380340357515 + 0j),
    "100010000110": (-0.0014842073044949633 + 0j),
    "100010100001": (0.002784753449849704 + 0j),
    "100010100010": (-0.0005755348992936581 + 0j),
    "100010100100": (0.00020450262296880706 + 0j),
    "100100000011": (0.0026739676813965505 + 0j),
    "100100000101": (-0.0001897232496828429 + 0j),
    "100100000110": (5.243203296635455e-05 + 0j),
    "100100100001": (0.0005586481007244873 + 0j),
    "100100100010": (0.00020450262296880514 + 0j),
    "100100100100": (0.00048055798189790397 + 0j),
    "101000001001": (-0.0003995905151859053 + 0j),
    "101000001010": (3.3923302022511554e-05 + 0j),
    "101000001100": (2.1873592193661752e-05 + 0j),
    "101000101000": (0.00017526636239952812 + 0j),
    "110000010001": (-0.0003995905151859029 + 0j),
    "110000010010": (3.392330202251112e-05 + 0j),
    "110000010100": (2.187359219366136e-05 + 0j),
    "110000110000": (0.00017526636239952772 + 0j),
}


def test_full_dist_lih_eigen():
    """Test full space solution against exact"""
    full_dist = {}
    for kk in range(2**OP.width):
        full_dist[bin(kk)[2:].zfill(OP.width)] = None

    S = fq.Subspace(full_dist)
    Hsub = fq.SubspaceHamiltonian(OP, S)

    # here we use starting vector of all ones to match phase with direct ans
    evals, evecs = spla.eigsh(Hsub, k=1, which="SA", v0=np.ones(len(S), dtype=complex))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_full_dist_lih_eigen_csr():
    """Test full space CSR solution against exact"""
    full_dist = {}
    for kk in range(2**OP.width):
        full_dist[bin(kk)[2:].zfill(OP.width)] = None

    S = fq.Subspace(full_dist)
    Hsub = fq.SubspaceHamiltonian(OP, S)
    M = Hsub.to_csr_linearoperator()

    # here we use starting vector of all ones to match phase with direct ans
    evals, evecs = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=complex))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_grnd_dist_lih_eigen():
    """Test grnd state space solution against exact"""
    S = fq.Subspace(GROUND_DIST)
    Hsub = fq.SubspaceHamiltonian(OP, S)

    # here we use starting vector of all ones to match phase with direct ans
    evals, evecs = spla.eigsh(Hsub, k=1, which="SA", v0=np.ones(len(S), dtype=complex))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)


def test_grnd_dist_lih_eigen_csr():
    """Test grnd space CSR solution against exact"""
    S = fq.Subspace(GROUND_DIST)
    Hsub = fq.SubspaceHamiltonian(OP, S)
    M = Hsub.to_csr_linearoperator()

    # here we use starting vector of all ones to match phase with direct ans
    evals, evecs = spla.eigsh(M, k=1, which="SA", v0=np.ones(len(S), dtype=complex))
    assert np.allclose(evals, GROUND_ENERGY, 1e-12)
