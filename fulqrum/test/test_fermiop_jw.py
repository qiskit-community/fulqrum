# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test Jordan-Wginer transformation of FermionicOperator terms"""
import numpy as np
import scipy.sparse as sp

from fulqrum import FermionicOperator
from fulqrum.utils import qubitoperator_to_matrix


# Tests compare JW transformations by looking at the resulting
# matrix element values compared to that from the JW transformation
# result using Qiskit Nature 0.7.2.  We do this because the operators
# themselves are not going to match because we use an extended
# alphabet.  Here, the matrices are given by the 3 arrays
# defining the CSR format.  Since Qiskit Nature need not be kept
# up to date with Qiskit itself, we hardcode the answers here
# rather than having Qiskit Nature as a dependency


def test_jw1():
    """Test single term JW conversion"""
    fop = FermionicOperator(3, [("-1+", [2, 1, 0], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([6])
    ans_indptr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    ans_data = np.array([1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw2():
    """Test single term JW conversion"""
    fop = FermionicOperator(3, [("-+", [2, 0], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([4, 6])
    ans_indptr = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2])
    ans_data = np.array([-1.0 + 0.0j, 1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw3():
    """Test single term JW conversion"""
    fop = FermionicOperator(3, [("-", [2], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([4, 5, 6, 7])
    ans_indptr = np.array([0, 1, 2, 3, 4, 4, 4, 4, 4])
    ans_data = np.array([1.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw4():
    """Test single term JW conversion"""
    fop = FermionicOperator(3, [("1", [2], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([4, 5, 6, 7])
    ans_indptr = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4])
    ans_data = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw5():
    """Test single term JW conversion"""
    fop = FermionicOperator(3, [("+", [2], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([0, 1, 2, 3])
    ans_indptr = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4])
    ans_data = np.array([1.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw6():
    """Test single term JW conversion"""
    fop = FermionicOperator(3, [("+-", [2, 1], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([2, 3])
    ans_indptr = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2])
    ans_data = np.array([1.0 + 0.0j, 1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw7():
    """Test single term JW conversion"""
    fop = FermionicOperator(3, [("+-", [1, 0], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([1, 5])
    ans_indptr = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    ans_data = np.array([1.0 + 0.0j, 1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw8():
    """Test single term JW conversion"""
    fop = FermionicOperator(5, [("+-", [4, 2], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([4, 5, 6, 7, 12, 13, 14, 15])
    ans_indptr = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            4,
            4,
            4,
            4,
            5,
            6,
            7,
            8,
            8,
            8,
            8,
            8,
        ]
    )
    ans_data = np.array(
        [
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            -1.0 + 0.0j,
            -1.0 + 0.0j,
            -1.0 + 0.0j,
            -1.0 + 0.0j,
        ]
    )

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw9():
    """Test single term JW conversion"""
    fop = FermionicOperator(5, [("---", [4, 2, 0], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([21, 23, 29, 31])
    ans_indptr = np.array(
        [
            0,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ]
    )
    ans_data = np.array([1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j])

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)


def test_jw10():
    """Test single term JW conversion"""
    fop = FermionicOperator(5, [("-+", [4, 2], 1)])
    op = fop.extended_jw_transformation()

    ans_indices = np.array([16, 17, 18, 19, 24, 25, 26, 27])
    ans_indptr = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            4,
            4,
            4,
            4,
            5,
            6,
            7,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ]
    )
    ans_data = np.array(
        [
            -1.0 + 0.0j,
            -1.0 + 0.0j,
            -1.0 + 0.0j,
            -1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
        ]
    )

    op_mat = sp.csr_array(qubitoperator_to_matrix(op))
    np.allclose(ans_indices, op_mat.indices, 1e-14)
    np.allclose(ans_indptr, op_mat.indptr, 1e-14)
    np.allclose(ans_data, op_mat.data)
