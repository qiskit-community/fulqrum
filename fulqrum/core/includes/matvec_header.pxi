# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector

include "base_header.pxi"

cdef extern from "../src/matvec.hpp":
    void omp_matvec(QubitOperator_t& ham,
                vector[unsigned char]& subspace,
                double complex * diag_vec,
                size_t width,
                size_t subspace_dim,
                int has_nonzero_diag,
                size_t bin_width,
                size_t * bin_ranges,
                const double complex * in_vec,
                double complex * out_vec)
