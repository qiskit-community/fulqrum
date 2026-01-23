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
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from ..core.qubit_operator cimport QubitOperator
from ..core.subspace cimport Subspace
from ..core.bitset cimport Bitset
from ..core.bitset cimport bitset_t
from ..core.linear_operator import SubspaceHamiltonian
from ..core.spmv cimport FulqrumSpMV

include "includes/simple_header.pxi"

import numpy as np


def ramps_simple_refinement(QubitOperator H, Subspace S, Bitset start, 
                            unsigned int max_recursion=3, double tol=1e-14):
    
    cdef Subspace out = Subspace([[start.to_string()]])
    Hsub = SubspaceHamiltonian(H, S)
    cdef FulqrumSpMV spmv = Hsub.spmv
    cdef double energy = simple_refinement(&spmv.oper.terms[0],
                                           spmv.subspace.subspace.bitstrings,
                                           out.subspace.bitstrings,
                                            spmv.diag_oper.terms,
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
    cdef list temp_out = []
    cdef size_t n
    cdef str bs
    for n in range(<size_t>out.size()):
        bs = out.get_n_th_bitstring(n)
        temp_out.append(bs)
    
    cdef Subspace final_out = Subspace([temp_out])
    
    return final_out, energy
