# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from fulqrum.core.subspace cimport Subspace

include "includes/base_header.pxi"


cdef class FulqrumSpMV:
    cdef QubitOperator_t oper
    cdef QubitOperator_t diag_oper
    cdef public Subspace subspace
    cdef public size_t subspace_dim
    cdef int num_threads
    cdef public size_t width
    cdef public size_t num_diag_terms
    cdef public size_t num_terms
    cdef public size_t bin_width
    cdef int has_nonzero_diag
    cdef double complex[::1] diag_vec
    cdef size_t * bin_ranges

    cdef void compute_diag_vector(self)
