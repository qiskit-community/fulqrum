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
"""Test basic core functionality"""

import numpy as np
import fulqrum as fq
from fulqrum import QubitOperator


def test_empty_qubitoperator():
    """Test empty QubitOperator"""
    N = 5
    qo = QubitOperator(N)
    assert qo.num_terms == 0


def test_identity():
    """Test identity QubitOperator"""
    N = 5
    qo = QubitOperator(N, [()])
    assert qo.num_terms == 1


def test_identity2():
    """Test identity QubitOperator 2"""
    N = 5
    qo = QubitOperator(N, [[]])
    assert qo.num_terms == 1


def test_coeff():
    """Test setting coeff for single identity operator"""
    N = 5
    qo = QubitOperator(N, [[1 + 2j]])
    assert qo.coefficients()[0] == 1 + 2j


def test_coeff2():
    """Test constant helper routiner"""
    N = 5
    qo2 = QubitOperator.from_constant(N, 1 + 2j)
    assert qo2.coefficients()[0] == 1 + 2j


def test_simple_multi_operators():
    """Test simple operator with non-identity weight"""
    N = 5
    qo = QubitOperator(N, [("X" * 5, range(5), 1)])
    assert qo.operators == [("X", kk) for kk in range(N)]


def test_simple_multi_weight():
    """Test simple operator with non-identity weight"""
    N = 5
    qo = QubitOperator(N, [("X" * 5, range(5), 1)])
    assert qo.weights() == [N]


def test_simple_inplace_add():
    """Test that inplace addition gives expected results"""
    N = 5
    op = QubitOperator(N)

    for kk in range(N):
        op += QubitOperator(N, [("Y", kk, 1 / (N + kk))])

    for kk in range(N):
        assert op[kk].operators == [("Y", kk)]
        assert op[kk].coefficients()[0] == 1.0 / (N + kk)


def test_simple_add():
    """Test that addition gives expected results"""
    N = 5
    op = QubitOperator(N)

    for kk in range(N):
        op = op + QubitOperator(N, [("0", kk, 1 / (N + kk))])

    for kk in range(N):
        assert op[kk].operators == [("0", kk)]
        assert op[kk].coefficients()[0] == 1.0 / (N + kk)


def test_simple_diagonal():
    """Verify diagonal operator returns True"""
    N = 25
    op = QubitOperator(N)
    diag_ops = ["I", "Z", "0", "1"]

    for kk in range(20):
        op += QubitOperator(N, [(diag_ops[kk % len(diag_ops)], kk, 1 / (N + kk))])

    assert op.is_diagonal()


def test_simple_diagonal2():
    """Verify non-diagonal operator returns False"""
    N = 25
    op = QubitOperator(N)
    diag_ops = ["I", "Z", "0", "1"]

    for kk in range(20):
        if kk:
            op += QubitOperator(N, [(diag_ops[kk % len(diag_ops)], kk, 1 / (N + kk))])
        # Add non-diagonal X operator when kk = 0
        else:
            op += QubitOperator(N, [("X", kk, 1 / (N + kk))])

    assert not op.is_diagonal()


def test_operator_diagonal_splitting():
    """Test operator diagonal splitting"""
    diag_ops = ["I", "X", "Z", "0", "Y", "1"]
    N = len(diag_ops)
    op = QubitOperator(N)

    for kk, oper in enumerate(diag_ops):
        op += QubitOperator(N, [(oper, 0, 1 / (N + kk))])
    diag, off_diag = op.split_diagonal()

    assert diag.num_terms == 4
    assert off_diag.num_terms == 2


def test_operator_identity_removal():
    """Test operator identity values and removal"""
    diag_ops = ["I", "X", "Z", "0", "Y", "I"]
    N = len(diag_ops)
    op = QubitOperator(N)

    for kk, oper in enumerate(diag_ops):
        op += QubitOperator(N, [(oper, 0, 1 / (N + kk))])

    assert abs(op.constant_energy() - 0.25757575757575757) < 1e-15
    new_op, _ = op.remove_constant_terms()
    assert new_op.num_terms == 4
    assert new_op.constant_energy() == 0


def test_operator_multiplication():
    """Test multiplication of QubitOperators by numbers"""
    N = 5
    qo = QubitOperator(N, [("X" * 5, range(N - 1, -1, -1), -1j)])
    new_qo = 5 * qo
    # operator ordering is switched due to internal sort
    assert new_qo.operators == [("X", 0), ("X", 1), ("X", 2), ("X", 3), ("X", 4)]
    assert new_qo.coefficients()[0] == -5j
    assert qo.coefficients()[0] == -1j
    qo *= 5
    assert qo.coefficients()[0] == -5j


def test_operator_sorting1():
    """Test simple operator sorting"""
    N = 5
    qo = QubitOperator(N, [("Z0+XY", [4, 0, 3, 1, 2], 1.0)])
    # operator ordering is switched due to internal sort
    assert qo.operators == [("0", 0), ("X", 1), ("Y", 2), ("+", 3), ("Z", 4)]


def test_operator_sorting2():
    """Test operator sorting with identity terms"""
    N = 5
    qo = QubitOperator(N, [("ZIIXY", [4, 0, 3, 1, 2], 1.0)])
    # operator ordering is switched due to internal sort
    assert qo.operators == [("X", 1), ("Y", 2), ("Z", 4)]


def test_operator_subtraction():
    """Test operator subtraction"""
    H = QubitOperator.from_label("ZZ")
    G = QubitOperator.from_label("XX", 5)
    qo = H - G
    assert qo[1].coefficients()[0] == -5.0


def test_proj_indices1():
    """Test projector indices are set properly #1"""
    op = fq.QubitOperator.from_label("1I0I0")
    assert np.allclose(op.proj_indices, [0, 2, 4])


def test_proj_indices2():
    """Test projector indices are set properly #2"""
    op = fq.QubitOperator(5, [("011", [0, 2, 3], 1)])
    assert np.allclose(op.proj_indices, [0, 2, 3])


def test_proj_indices3():
    """Validate no proj indices for term"""
    op = fq.QubitOperator.from_label("IZXY+-")
    assert not any(op.proj_indices)


def test_proj_indices4():
    """Validate no proj indices for empty term"""
    op = fq.QubitOperator(5)
    assert not any(op.proj_indices)


def test_is_real():
    """Is real is true for small imag <= 1e-12"""
    op = fq.QubitOperator.from_label("II++")
    op = (1 + 1e-14j) * fq.QubitOperator.from_label("II--")
    op.set_type(2)
    assert op.is_real()


def test_is_real2():
    """Is real is false for imag > 1e-12"""
    op = fq.QubitOperator.from_label("II++")
    op = (1 + 1e-11j) * fq.QubitOperator.from_label("II--")
    op.set_type(2)
    assert not op.is_real()


def test_is_real3():
    """Is real is false Pauli-based operators"""
    op = fq.QubitOperator.from_label("IIXX")
    op = (1 + 1e-11j) * fq.QubitOperator.from_label("IIYY")
    assert not op.is_real()


def test_qubitop_iter():
    """Verify that QubitOperator iterates properly"""
    op = fq.QubitOperator.from_label("IIXXII")
    op += fq.QubitOperator.from_label("YIXXIY")
    op += fq.QubitOperator.from_label("ZZZZZZ")

    for idx, item in enumerate(op):
        assert item.operators == op[idx].operators
