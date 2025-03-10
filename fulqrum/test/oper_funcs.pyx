# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from fulqrum.core.qubit_operator cimport QubitOperator

include "../core/includes/base_header.pxi"
include "../core/includes/operators_header.pxi"


def nonzero_extended_value_wrapper(QubitOperator ham, unsigned char[::1] bits):
    """Wrapper that helps with testing the nonzero_extended_value routine

    Input operator should have a single term only
    
    Parameters:
        input (str): Input string
    
    Returns:
        ndarray: Array of type uintp
    """
    if ham.oper.terms.size() != 1:
        raise Exception('Single term operator only')
    cdef OperatorTerm_t * term = &ham.oper.terms[0]
    return nonzero_extended_value(term, &bits[0], ham.width)
