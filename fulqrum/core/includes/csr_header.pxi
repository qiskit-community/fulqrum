# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
include "base_header.pxi"

ctypedef long long int64

ctypedef fused fused_int:
    int
    int64


cdef extern from "../src/csr.hpp":
    void csr_matvec[T](const double complex * data, const T * indices, const T * indptr,
                      const double complex * vec, double complex * out, size_t nrows) nogil


    void csr_builder[T](QubitOperator_t& ham, vector[unsigned char]& subspace,
                        double complex * diag_vec, size_t width, size_t subspace_dim,
                        int has_nonzero_diag, size_t bin_width, size_t * bin_ranges,
                        T * indptr, T * indices, double complex * data,
                        int compute_values) nogil
