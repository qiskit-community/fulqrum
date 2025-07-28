# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t
include "base_header.pxi"


cdef extern from "../src/csr2.hpp":

    void csr_matrix_builder2[T, U](const OperatorTerm_t * terms,
                              const vector[bitset_t]& subspace,
                              const U * diag_vec,
                              size_t width,
                              size_t subspace_dim,
                              int has_nonzero_diag,
                              size_t bin_width,
                              const size_t * bin_ranges,
                              const size_t * group_ptrs,
                              const size_t * group_ladder_ptrs,
                              unsigned int * group_rowint_length,
                              const vector[vector[unsigned int]]& group_offdiag_inds,
                              size_t num_groups,
                              unsigned int ladder_offset,
                              T * indptr,
                              T * indices,
                              U * data,
                              int compute_values) nogil
