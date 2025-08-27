# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper

include "base_header.pxi"

cdef extern from "../src/diag.hpp":
    void compute_diag_vector[T](const BitsetHashMapWrapper& data,
                                T * diag_vec,
                                const QubitOperator_t& diag_oper,
                                const unsigned int width,
                                const size_t subspace_dim) nogil
