# Fulqrum
# Copyright (C) 2024, IBM
cimport cython
from libcpp.vector cimport vector

include "includes/base_header.pxi"

ctypedef long long int64_t
ctypedef fused fused_int:
    int
    int64_t


cdef void csr_matrix_builder(QubitOperator_t& ham,
                            vector[unsigned char]& subspace,
                            double complex * diag_vec,
                            size_t width,
                            size_t subspace_dim,
                            int has_nonzero_diag,
                            size_t bin_width,
                            size_t * bin_ranges,
                            size_t * group_ptrs,
                            size_t num_groups,
                            fused_int * indptr,
                            fused_int * indices,
                            double complex * data,
                            int compute_values) noexcept nogil
