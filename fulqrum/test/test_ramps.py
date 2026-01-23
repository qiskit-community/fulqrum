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
"""Test method(s) in linear operator that are not covered in other tests"""
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla

from fulqrum import FermionicOperator, Subspace, SubspaceHamiltonian
from fulqrum.ramps import ramps_simple_refinement


_path = Path(__file__).parent / "data/lih.json"
FOP = FermionicOperator.from_json(_path)
NEW_OP = FOP.extended_jw_transformation()
NEW_OP, _ = NEW_OP.remove_constant_terms()

# Build full Hilbert space
DIST = []
for kk in range(2**12):
    DIST.append(bin(kk)[2:].zfill(NEW_OP.width))

S = Subspace([DIST])
HSUB = SubspaceHamiltonian(NEW_OP, S)
EXACT_ENERGY = spla.eigsh(HSUB.to_csr_linearoperator_fast(), k=1, which="SA")[0][0]


def test_ramps_simple_refine_subspace_dim():
    """Verify size of RAMPS subspace dim for LiH"""
    diag = HSUB.diagonal_vector()
    min_idx = np.where(diag == diag.min())[0][0]

    start = S[min_idx]
    out, _ = ramps_simple_refinement(NEW_OP, S, start)
    assert out.size() == 69


def test_ramps_simple_refine_accuracy():
    """Verify sRAMPS energy is close to exact answer for LiH"""
    diag = HSUB.diagonal_vector()
    min_idx = np.where(diag == diag.min())[0][0]

    start = S[min_idx]
    out, _ = ramps_simple_refinement(NEW_OP, S, start)

    Hsub_small = SubspaceHamiltonian(NEW_OP, out)
    approx_energy = spla.eigsh(
        Hsub_small.to_csr_linearoperator_fast(), k=1, which="SA"
    )[0][0]

    assert abs((approx_energy - EXACT_ENERGY) / EXACT_ENERGY) < 1e-14
