# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.subspace cimport Subspace
from fulqrum.core.bitset cimport Bitset

include "includes/simple_header.pxi"


def ramps_simple_refinement(QubitOperator H, Subspace S, Bitset start, 
                            unsigned int max_recursion=4, double tol=1e-14):
    
    cdef Subspace out = Subspace({start.to_string(): 0})
    cdef QubitOperator diag_op, off_op
    diag_op, off_op = H.split_diagonal()
    
    return 0
