# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
include "base_header.pxi"


cdef extern from "../src/csr.hpp":

    void csr_matrix_builder[T](OperatorTerm_t * terms,
                              vector[unsigned char]& subspace,
                              const double complex * diag_vec,
                              size_t width,
                              size_t subspace_dim,
                              int has_nonzero_diag,
                              size_t bin_width,
                              const size_t * bin_ranges,
                              const size_t * group_ptrs,
                              size_t num_groups,
                              T * indptr,
                              T * indices,
                              double complex * data,
                              int compute_values) nogil
