# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper

include "../../core/includes/base_header.pxi"


cdef extern from "../src/simple.hpp":

    double simple_refinement(const vector[OperatorTerm_t]& diag_terms,
                          const vector[OperatorTerm_t]& off_terms,
                          const bitset_t start,
                          const BitsetHashMapWrapper& subspace,
                          BitsetHashMapWrapper& out_subspace,
                          unsigned max_recursion,
                          const size_t * group_ptrs,
                          const size_t * group_ladder_ptrs,
                          unsigned int * group_rowint_length,
                          const vector[vector[unsigned int]]& group_offdiag_inds,
                          size_t num_groups,
                          unsigned int ladder_offset,
                           ) nogil
