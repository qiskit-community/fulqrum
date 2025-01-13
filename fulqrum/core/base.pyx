# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp cimport bool
from fulqrum.core.base cimport OperatorTerm
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool diagonal_term(OperatorTerm * term):
    """Check if term is diagonal in computational basis

    Returns:
        bool: True if diagonal
    """
    cdef size_t kk
    for kk in range(term.values.size()):
        if term.values[kk] > 2:
            return False
    return True
