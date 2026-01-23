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
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

import fulqrum as fq


_path = Path(__file__).parent / "data/lih.json"
FOP = fq.FermionicOperator.from_json(_path)
OP = FOP.extended_jw_transformation()


def test_interpret_renormalization():
    """Validate that renormalization returns vectors with probabilities sum to one"""
    full_dist = []
    for kk in range(2**OP.width):
        full_dist.append(bin(kk)[2:].zfill(OP.width))

    S = fq.Subspace([full_dist])
    Hsub = fq.SubspaceHamiltonian(OP, S)

    x0 = np.ones(len(S), dtype=Hsub.dtype)
    _, evecs = spla.eigsh(Hsub, k=1, which="SA", v0=x0)

    grnd_state = Hsub.interpret_vector(evecs, atol=1e-14)
    assert abs(sum(np.array(list(grnd_state.values())) ** 2) - 1.0) < 1e-14

    grnd_state = Hsub.interpret_vector(evecs, atol=1e-6)
    assert abs(sum(np.array(list(grnd_state.values())) ** 2) - 1.0) < 1e-14

    grnd_state = Hsub.interpret_vector(evecs, atol=1e-3)
    assert abs(sum(np.array(list(grnd_state.values())) ** 2) - 1.0) < 1e-14
