# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t

include "base_header.pxi"

cdef extern from "../src/matvec2.hpp":
    void omp_matvec2(QubitOperator_t& ham,
                vector[bitset_t]& subspace,
                double complex * diag_vec,
                size_t width,
                size_t subspace_dim,
                int has_nonzero_diag,
                size_t bin_width,
                size_t * bin_ranges,
                size_t * group_ptrs,
                size_t * group_ladder_ptrs,
                const vector[vector[unsigned int]]& group_offdiag_inds,
                size_t num_groups,
                const double complex * in_vec,
                double complex * out_vec) nogil
