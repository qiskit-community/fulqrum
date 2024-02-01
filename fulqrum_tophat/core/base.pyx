# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp cimport bool
from fulqrum_tophat.core.base cimport OperatorTerm
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool diagonal_term(OperatorTerm * term):
    """Check if term is diagonal in computational basis

    Returns:
        bool: True if diagonal
    """
    cdef size_t kk
    for kk in range(term.operators.size()):
        if term.operators[kk].second > 2:
            return False
    return True
