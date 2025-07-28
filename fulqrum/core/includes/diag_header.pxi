# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t

include "base_header.pxi"

cdef extern from "../src/diag.hpp":
    void compute_diag_vector[T](const vector[bitset_t]& data,
                                T * diag_vec,
                                const QubitOperator_t& diag_oper,
                                const unsigned int width,
                                const size_t subspace_dim) nogil
