# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper

include "../../core/includes/base_header.pxi"


cdef extern from "../src/simple.hpp":

    double simple_refinement[U](const OperatorTerm_t * terms,
                             const BitsetHashMapWrapper &subspace,
                             BitsetHashMapWrapper& out_subspace,
                             const U * diag_vec,
                             const unsigned int width,
                             const size_t subspace_dim,
                             const int has_nonzero_diag,
                             const size_t * group_ptrs,
                             const size_t * group_ladder_ptrs,
                             unsigned int * group_rowint_length,
                             vector[vector[unsigned int]]& group_offdiag_inds,
                             const size_t num_groups,
                             const unsigned int ladder_offset,
                             unsigned int max_recursion,
                             double tol
                               ) nogil
