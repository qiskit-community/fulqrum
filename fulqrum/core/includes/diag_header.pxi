# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector

include "base_header.pxi"

cdef extern from "../src/diag.hpp":
    void compute_diag_vector(const unsigned char * data,
                             double complex * diag_vec,
                             QubitOperator_t& diag_oper,
                             size_t width,
                             size_t subspace_dim) nogil
