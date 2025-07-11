# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.vector cimport vector
from fulqrum.core.subspace cimport Subspace

include "includes/base_header.pxi"


cdef class FulqrumSpMV:
    cdef QubitOperator_t oper
    cdef QubitOperator_t diag_oper
    cdef public Subspace subspace
    cdef public size_t subspace_dim
    cdef public unsigned int width
    cdef public size_t num_diag_terms
    cdef public size_t num_terms
    cdef public unsigned int bin_width
    cdef int has_nonzero_diag
    cdef double complex[::1] diag_vec
    cdef size_t[::1] group_ptrs
    cdef int num_groups
    cdef size_t * bin_ranges
    cdef vector[vector[unsigned int]] group_offdiag_inds

    cdef void compute_diag_vector(self)
