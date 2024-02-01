# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython

from libcpp.pair cimport pair
from libcpp.string cimport string
from cython.operator cimport dereference
from fulqrum_tophat.core.qubit_operator cimport QubitOperator, OPER_ELEMS
from fulqrum_tophat.core.base cimport OperatorTerm

ctypedef pair[string, double complex] str_complex_pair


@cython.exceptval(check=False)
@cython.boundscheck(False)
cdef void matrix_elem_by_row(const OperatorTerm * term, 
                             const string * row,
                             str_complex_pair * out) nogil:
    """Returns the nonzero column bit-string and matrix element value
    for a given row bit-string of a OperatorTerm

    Parameters:
        term (OperatorTerm *): Pointer to OperatorTerm
        row (string *): Pointer to string representing row
        out (str_complex_par *): Point to pair of string and double complex
    """
    # make a copy of row
    cdef string col = dereference(row)
    cdef size_t kk, jj
    cdef size_t num_qubits = row.size()
    cdef double complex val = 1.0
    cdef size_t idx
    for kk in range(term.operators.size()):
        # Off-diagonal element
        idx = num_qubits-term.operators[kk].first-1
        if term.operators[kk].second > 2:
            # Row element is zero, so flip the column bit to one
            if row.at(idx) == 48:
                col[idx] = 49
                val *= OPER_ELEMS[4*term.operators[kk].second+1]
            # Row element is one, so flip the column to zero
            else:
                col[idx] = 48
                val *= OPER_ELEMS[4*term.operators[kk].second+2]
        # Diagonal element
        else:
            # Row element is zero
            if row.at(idx) == 48:
                val *= OPER_ELEMS[4*term.operators[kk].second]
            # Row element is one
            else:
                val *= OPER_ELEMS[4*term.operators[kk].second+3]
    # Output column bitstring
    out.first = col
    # Output value for matrix element
    out.second = term.coeff*val
