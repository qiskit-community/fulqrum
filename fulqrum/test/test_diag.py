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
"""Test diagonal vector computation"""

from pathlib import Path
import numpy as np

import fulqrum as fq
from fulqrum.utils.io import json_to_dict


def test_diag_fast_mode():
    """Verify fast mode and default give same diagonal"""
    op_path = Path(__file__).parent / "data/ch4_dimer_jw.json.xz"
    op = fq.QubitOperator.from_json(op_path)

    dist_path = Path(__file__).parent / "data/dimer_subspace.json.xz"
    dist = json_to_dict(dist_path)
    S = fq.Subspace([list(dist.keys())])
    Hsub = fq.SubspaceHamiltonian(op, S)
    diag1 = Hsub.diagonal_vector()  # This is fast mode
    diag2 = Hsub.diagonal_vector(disable_fast_mode=True)  # This is regular mode
    np.allclose(diag1, diag2, 1e-14)
