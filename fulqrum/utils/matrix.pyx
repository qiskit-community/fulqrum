# This code is a Qiskit project.
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
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.string cimport string
from fulqrum.exceptions import FulqrumError
from fulqrum.core.qubit_operator cimport QubitOperator
import numpy as np

include "fulqrum/core/includes/base_header.pxi"
include "fulqrum/core/includes/converters.pxi"


X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

I = np.array([[1, 0], [0, 1]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

G = np.array([[1, 0], [0, 0]], dtype=complex)

O = np.array([[0, 0], [0, 1]], dtype=complex)

P = np.array([[0, 0], [1, 0]], dtype=complex)

M = np.array([[0, 1], [0, 0]], dtype=complex)

str_matrix_convert = {"I": I, "Z": Z, "0": G, "1": O, "X": X, "Y": Y, "+": P, "-": M}


def kron_str(operator_string):
    """Performs the kron over a string consisting of qubit operators"""
    mat = str_matrix_convert[operator_string[0].upper()]
    cdef unsigned int kk
    cdef unsigned int str_len = len(operator_string)
    for kk in range(1, str_len):
        mat = np.kron(mat, str_matrix_convert[operator_string[kk].upper()])
    return mat


@cython.boundscheck(False)
def qubitoperator_to_matrix(QubitOperator op):
    """Convert a QubitOperator to a dense matrix

    Parameters:
        op (QubitOperator): Input operator

    Returns:
        ndarray: Complex valued NumPy array

    Raises:
        FulqrumError: Supports conversion for <= 10 qubits only
    """
    if op.width > 10:
        raise FulqrumError('Casting to dense matrix works for <= 10 qubits only')
    A = np.zeros((2**op.width,)*2, dtype=complex)
    cdef int width = op.width
    cdef size_t num_terms = op.oper.terms.size()
    cdef OperatorTerm_t * term
    cdef size_t kk, jj
    cdef list str_list
    for kk in range(num_terms):
        term = &op.oper.terms[kk]
        str_list = ['I'] * width
        for jj in range(term.indices.size()):
            str_list[term.indices[jj]] = IND_TO_STR[term.values[jj]]
        A += term.coeff * kron_str(str_list[::-1])
    return A
