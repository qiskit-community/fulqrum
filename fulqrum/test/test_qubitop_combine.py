# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test combining qubit terms"""
from pathlib import Path
import numpy as np
from fulqrum import FermionicOperator
from fulqrum.utils import qubitoperator_to_matrix


def test_combining_h2_terms():
    """Validate combining repeat qubitop terms yields same num_terms and numberic operator"""
    path = Path(__file__).parent / "data/H2.json"
    fop = FermionicOperator.from_json(path)
    assert fop.num_terms == 36
    op = fop.extended_jw_transformation()
    assert op.num_terms == 28
    new_op = op.combine_repeated_terms()
    assert new_op.num_terms == 14
    mat1 = qubitoperator_to_matrix(op)
    mat2 = qubitoperator_to_matrix(new_op)
    assert np.allclose(mat1, mat2, 1e-14)
