# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper

include "../../core/includes/base_header.pxi"


cdef extern from "../src/simple.hpp":

    int simple_refinement(const OperatorTerm_t * diag_terms,
                          const OperatorTerm_t * off_terms,
                          const BitsetHashMapWrapper& subspace,
                          BitsetHashMapWrapper& out_subspace,
                         ) nogil
