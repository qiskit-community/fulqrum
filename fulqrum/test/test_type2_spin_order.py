# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=no-name-in-module
"""Tests for type-2 groups that carry no alpha and beta order. See Issue # 91"""

import itertools
from pathlib import Path

import numpy as np
import pytest

import fulqrum as fq

PATH = str(Path(__file__).parent) + "/data/"

LADDER_WIDTHS = [1, 2, 4]
KERNELS = ["matvec", "csr", "csr_fast"]

DOUBLE_EXCITATION_WIDTH = 28
DOUBLE_EXCITATION_DETS = [
    "0000000000000111111111111111",
    "0000000000011111111111111100",
]

SYNTHETIC_WIDTH = 6


def _double_excitation_operator(coeff=1.0):
    """Builds the qubit operator of the two-particle group."""
    fermionic_op = fq.FermionicOperator(
        DOUBLE_EXCITATION_WIDTH,
        [
            ("+-+-", [16, 0, 15, 1], coeff),
            ("+-+-", [1, 15, 0, 16], np.conj(coeff)),
        ],
    )
    return fermionic_op.extended_jw_transformation()


def _reference_matrix(op, subspace):
    """Builds the dense reference from ``matrix_element``."""
    strings = []
    for index in range(len(subspace)):
        value = subspace.get_n_th_bitstring(index)
        strings.append(value.decode() if isinstance(value, bytes) else value)

    out = np.zeros((len(strings), len(strings)), dtype=complex)
    for row, row_str in enumerate(strings):
        for col, col_str in enumerate(strings):
            out[row, col] = op.matrix_element(row_str, col_str)
    return out


def _kernel_matrix(hsub, kernel):
    """Builds the dense matrix of one kernel."""
    if kernel == "csr":
        return hsub.to_csr_linearoperator().matrix.toarray()
    if kernel == "csr_fast":
        return hsub.to_csr_linearoperator_fast().matrix.toarray()
    dim = hsub.shape[0]
    basis = np.eye(dim, dtype=hsub.dtype)
    return np.column_stack([hsub @ basis[:, idx] for idx in range(dim)])


def _check_kernel(op, subspace, kernel, ladder_width, monkeypatch, block_size=None):
    """Compares one kernel against the ``matrix_element`` reference."""
    monkeypatch.setenv("FQ_LADDER_WIDTH", str(ladder_width))
    if block_size is not None:
        monkeypatch.setenv("FQ_BLK", str(block_size))

    reference = _reference_matrix(op, subspace)
    hsub = fq.SubspaceHamiltonian(op, subspace)
    result = _kernel_matrix(hsub, kernel)
    assert np.allclose(result, reference, atol=1e-12), (
        f"{kernel} differs from matrix_element at ladder_width={ladder_width}"
    )
    return reference


def _offdiag_connections(reference):
    """Counts the non-zero off-diagonal elements of a reference matrix."""
    off_diagonal = reference - np.diag(np.diag(reference))
    return int(np.count_nonzero(np.abs(off_diagonal) > 1e-12))


def _connected_subspace(op, seed_strings, limit):
    """Grows a subspace from seed determinants along the group flip indices."""
    _, off = op.split_diagonal()
    off.group_sort()
    group_inds = off.group_offdiag_indices()
    width = len(seed_strings[0])
    dets = set(seed_strings)
    for seed in seed_strings:
        for indices in group_inds:
            bits = list(seed)
            for index in indices:
                # Bit position 0 is the rightmost character.
                position = width - 1 - int(index)
                bits[position] = "0" if bits[position] == "1" else "1"
            dets.add("".join(bits))
            if len(dets) >= limit:
                return sorted(dets)
    return sorted(dets)


def _random_spinless_operator(num_modes, num_terms, seed):
    """Builds a Hermitian spinless two-body operator with no spin order."""
    rng = np.random.default_rng(seed)
    terms = []
    for _ in range(num_terms):
        p, q, r, s = (int(x) for x in rng.choice(num_modes, size=4, replace=False))
        coeff = float(rng.normal())
        terms.append(("+-+-", [p, q, r, s], coeff))
        # The Hermitian conjugate of a_p^+ a_q a_r^+ a_s is a_s^+ a_r a_q^+ a_p.
        terms.append(("+-+-", [s, r, q, p], coeff))
    return fq.FermionicOperator(num_modes, terms).extended_jw_transformation()


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("ladder_width", LADDER_WIDTHS)
def test_double_excitation_real_coefficient(kernel, ladder_width, monkeypatch):
    """The two determinants connect, and every kernel finds the element."""
    op = _double_excitation_operator()
    subspace = fq.Subspace([list(DOUBLE_EXCITATION_DETS)])
    reference = _check_kernel(op, subspace, kernel, ladder_width, monkeypatch)

    assert _offdiag_connections(reference) == 2


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("ladder_width", LADDER_WIDTHS)
def test_double_excitation_complex_coefficient(kernel, ladder_width, monkeypatch):
    op = _double_excitation_operator(coeff=0.5 + 0.25j)
    subspace = fq.Subspace([list(DOUBLE_EXCITATION_DETS)])
    reference = _check_kernel(op, subspace, kernel, ladder_width, monkeypatch)

    assert _offdiag_connections(reference) == 2


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("ladder_width", LADDER_WIDTHS)
def test_random_spinless_operator(kernel, ladder_width, monkeypatch):
    op = _random_spinless_operator(num_modes=12, num_terms=14, seed=7)
    seeds = ["000111000111", "010101101010", "111000000111"]
    subspace = fq.Subspace([_connected_subspace(op, seeds, limit=90)])
    reference = _check_kernel(op, subspace, kernel, ladder_width, monkeypatch)
    assert _offdiag_connections(reference) > 0


@pytest.mark.parametrize("kernel", KERNELS)
def test_random_spinless_small_block(kernel, monkeypatch):
    op = _random_spinless_operator(num_modes=12, num_terms=14, seed=7)
    subspace = fq.Subspace([_connected_subspace(op, ["000111000111"], limit=64)])
    reference = _check_kernel(op, subspace, kernel, 2, monkeypatch, block_size=3)
    assert _offdiag_connections(reference) > 0


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("ladder_width", LADDER_WIDTHS)
def test_lih_half_strings(kernel, ladder_width, monkeypatch):
    """LiH in half-string mode, which has a clean alpha and beta split."""
    op = fq.FermionicOperator.from_json(PATH + "lih.json").extended_jw_transformation()
    half_width = op.width // 2
    halves = []
    for positions in itertools.combinations(range(half_width), 2):
        bits = ["0"] * half_width
        for position in positions:
            bits[half_width - 1 - position] = "1"
        halves.append("".join(bits))
    halves = sorted(halves)[:6]

    subspace = fq.Subspace([list(halves), list(halves)])
    reference = _check_kernel(op, subspace, kernel, ladder_width, monkeypatch)
    assert _offdiag_connections(reference) > 0


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("ladder_width", LADDER_WIDTHS)
def test_lih_full_strings(kernel, ladder_width, monkeypatch):
    """LiH on determinants that carry no alpha and beta order."""
    op = fq.FermionicOperator.from_json(PATH + "lih.json").extended_jw_transformation()
    seeds = ["000111000111", "011001100110"]
    subspace = fq.Subspace([_connected_subspace(op, seeds, limit=70)])
    reference = _check_kernel(op, subspace, kernel, ladder_width, monkeypatch)
    assert _offdiag_connections(reference) > 0


# Each entry is Hermitian, thus it is a Hamiltonian that fulqrum accepts.
SYNTHETIC_CASES = [
    (
        "paired_double_excitation",
        [("--++", [0, 1, 4, 5], 1.0), ("++--", [0, 1, 4, 5], 1.0)],
    ),
    (
        "paired_single_excitations",
        [("+-+-", [0, 1, 4, 5], 1.0), ("-+-+", [0, 1, 4, 5], 1.0)],
    ),
    (
        "unpaired_one_sided",
        [("++--", [0, 1, 2, 3], 1.0), ("--++", [0, 1, 2, 3], 1.0)],
    ),
    (
        "unpaired_in_complex_operator",
        [
            ("++--", [0, 1, 2, 3], 1.0),
            ("--++", [0, 1, 2, 3], 1.0),
            ("+-", [4, 5], 0.5 + 0.25j),
            ("-+", [4, 5], 0.5 - 0.25j),
        ],
    ),
    # Standard. A projector rules the group out of the direct path.
    (
        "projector_in_group",
        [("--++1", [0, 1, 4, 5, 3], 1.0), ("++--1", [0, 1, 4, 5, 3], 1.0)],
    ),
    # Standard. The flips at 0, 2, 3 and 5 need a Z at 1 and at 4.
    (
        "missing_z_string",
        [("+-+-", [0, 2, 3, 5], 1.0), ("-+-+", [0, 2, 3, 5], 1.0)],
    ),
    # Standard. The flips at 0, 1, 4 and 5 need no Z, so the Z at 2 is extra.
    (
        "extra_z_string",
        [("+-Z+-", [0, 1, 2, 4, 5], 1.0), ("-+Z-+", [0, 1, 2, 4, 5], 1.0)],
    ),
    # Standard. Two terms of the group need the same row pattern.
    (
        "repeated_term",
        [
            ("--++", [0, 1, 4, 5], 1.0),
            ("--++", [0, 1, 4, 5], 1.0),
            ("++--", [0, 1, 4, 5], 1.0),
            ("++--", [0, 1, 4, 5], 1.0),
        ],
    ),
    # Standard. The group holds two different coefficients.
    (
        "mixed_coefficients",
        [
            ("+-+-", [0, 1, 4, 5], 1.0),
            ("-+-+", [0, 1, 4, 5], 1.0),
            ("--++", [0, 1, 4, 5], 0.25),
            ("++--", [0, 1, 4, 5], 0.25),
        ],
    ),
    # Standard, two flips. Both operators create a particle, therefore the group
    # does not conserve the particle number.
    (
        "non_conserving_two_flip",
        [("++", [0, 1], 1.0), ("--", [0, 1], 1.0)],
    ),
]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "label,terms", SYNTHETIC_CASES, ids=[c[0] for c in SYNTHETIC_CASES]
)
def test_synthetic_type2_operators(kernel, label, terms, monkeypatch):
    """Synthetic type-2 operators over the full space of six qubits."""
    op = fq.QubitOperator(SYNTHETIC_WIDTH, terms)
    op.set_type(2)
    full_space = fq.Subspace(
        [[bin(value)[2:].zfill(SYNTHETIC_WIDTH) for value in range(2**SYNTHETIC_WIDTH)]]
    )
    reference = _check_kernel(op, full_space, kernel, 2, monkeypatch)
    assert _offdiag_connections(reference) > 0, label


def test_kernels_agree_on_molecular_operator(monkeypatch):
    monkeypatch.setenv("FQ_LADDER_WIDTH", "2")
    op = fq.QubitOperator.from_json(PATH + "ch4_dimer_jw.json.xz")
    dist = fq.utils.io.json_to_dict(PATH + "dimer_subspace.json.xz")
    subspace = fq.Subspace([list(dist.keys())[:8000]])
    hsub = fq.SubspaceHamiltonian(op, subspace)

    csr = hsub.to_csr_linearoperator().matrix
    csr_fast = hsub.to_csr_linearoperator_fast().matrix
    assert csr.nnz > 0
    assert csr.nnz == csr_fast.nnz
    assert np.allclose(csr.indptr, csr_fast.indptr)
    assert np.allclose(csr.indices, csr_fast.indices)
    assert np.allclose(csr.data, csr_fast.data)

    rng = np.random.default_rng(3)
    vector = rng.standard_normal(len(subspace)).astype(hsub.dtype)
    assert np.allclose(hsub @ vector, csr @ vector, atol=1e-10)
