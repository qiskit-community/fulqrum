# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.subspace cimport Subspace
from fulqrum.core.bitset cimport Bitset
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.linear_operator import SubspaceHamiltonian
from fulqrum.core.spmv cimport FulqrumSpMV

include "includes/simple_header.pxi"

import numpy as np


def ramps_simple_refinement(QubitOperator H, Subspace S, Bitset start, 
                            unsigned int max_recursion=3, double tol=1e-14):
    
    cdef Subspace out = Subspace({start.to_string(): 0})
    Hsub = SubspaceHamiltonian(H, S)
    cdef FulqrumSpMV spmv = Hsub.spmv
    spmv.compute_diag_vector()
    cdef double energy = simple_refinement[double](&spmv.oper.terms[0],
                                           spmv.subspace.subspace.bitstrings,
                                           out.subspace.bitstrings,
                                            &spmv.real_diag_vec[0],
                                            spmv.width,
                                            spmv.subspace_dim,
                                            spmv.has_nonzero_diag,
                                            &spmv.group_ptrs[0],
                                            &spmv.group_ladder_ptrs[0],
                                            &spmv.group_rowint_length[0],
                                            spmv.group_offdiag_inds,
                                            spmv.num_groups,
                                            spmv.ladder_offset,
                                            max_recursion,
                                            tol)    
    
    # This is a temp workaround for issues with iteratively expanded
    # subspaces.  This should be removed once those are resolved
    cdef dict temp_out = {}
    cdef size_t n
    cdef str bs
    for n in range(out.size()):
        bs = out.get_n_th_bitstring(n)
        temp_out[bs] = 1
    
    cdef Subspace final_out = Subspace(temp_out)
    
    return final_out, energy
