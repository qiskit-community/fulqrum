# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.bitset cimport Bitset, bitset_t

include "../core/includes/base_header.pxi"
include "../core/includes/bitset_utils_header.pxi"


def nonzero_extended_value_wrapper(QubitOperator ham, Bitset bits):
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
    return passes_proj_validation(term, bits.bits)
