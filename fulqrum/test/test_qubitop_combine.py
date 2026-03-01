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
"""Test combining qubit terms"""

from pathlib import Path
import numpy as np
from fulqrum import QubitOperator, FermionicOperator


def test_combining_terms1():
    op = QubitOperator.from_label("IZYXI")
    op += QubitOperator.from_label("IZYXI")
    op += QubitOperator.from_label("IZYXI")
    op += QubitOperator.from_label("IZYXI")
    op += QubitOperator.from_label("IZYXI")
    op += QubitOperator.from_label("IIIII")
    op += QubitOperator.from_label("I0YXI")
    new_op = op.combine_repeated_terms()
    assert new_op.num_terms == 3
    assert np.allclose(new_op.weights(), [0, 3, 3])
    assert new_op[1].coefficients()[0] == 5.0


def test_combining_h2_terms():
    """Validate combining repeat qubitop terms yields same num_terms and numeric operator"""
    path = Path(__file__).parent / "data/h2.json"
    fop = FermionicOperator.from_json(path)
    assert fop.num_terms == 36
    op = fop.extended_jw_transformation()
    assert op.num_terms == 14
