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
from libcpp.vector cimport vector
from .subspace cimport Subspace

include "includes/base_header.pxi"
include "includes/types.pxi"

cdef class FulqrumSpMV:
    cdef QubitOperator_t oper
    cdef QubitOperator_t diag_oper
    cdef public Subspace subspace
    cdef public size_t subspace_dim
    cdef public unsigned int width
    cdef public size_t num_diag_terms
    cdef public size_t num_terms
    cdef public int has_nonzero_diag
    cdef int init_diag
    cdef double complex[::1] complex_diag_vec
    cdef double[::1] real_diag_vec # Need to split this due to Cython fused type limitation
    cdef size_t[::1] group_ptrs
    cdef size_t[::1] group_ladder_ptrs
    cdef unsigned int[::1] group_rowint_length
    cdef int num_groups
    cdef public int is_real
    cdef unsigned int ladder_offset
    cdef vector[vector[unsigned int]] group_offdiag_inds

    cpdef int compute_diag_vector(self)
