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
import pytest
from pathlib import Path

import numpy as np
import fulqrum as fq
from fulqrum.exceptions import FulqrumError

_path = Path(__file__).parent / "data/lih.json"
FOP = fq.FermionicOperator.from_json(_path)
OP = FOP.extended_jw_transformation()


def test_subspacehamiltonian_different_widths():
    """Hamiltonian and subspace with different widths raises"""
    S = fq.Subspace()
    Hsub = fq.SubspaceHamiltonian(OP, S)
    with pytest.raises(FulqrumError) as _:
        Hsub.diagonal_vector()


def test_subspacehamiltonian_update_subspace():
    """Test that updating a subspace works"""
    S = fq.Subspace()
    S2 = fq.Subspace([["1" * 12]])
    Hsub = fq.SubspaceHamiltonian(OP, S)
    Hsub.update_subspace(S2)
    diag = Hsub.diagonal_vector()
    assert np.allclose(diag, np.array([1.77959097]))


def test_subspacehamiltonian_init_no_subspace():
    """Can initialize a SubspaceHamiltonian with no subspace"""
    Hsub = fq.SubspaceHamiltonian(OP)
    assert Hsub.spmv.subspace_dim == 0
