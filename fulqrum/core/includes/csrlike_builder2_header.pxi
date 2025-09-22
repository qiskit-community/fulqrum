# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from libcpp.deque cimport deque
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper
include "base_header.pxi"
include "csrlike_header.pxi"


cdef extern from "../src/csrlike_builder2.hpp":

    void csrlike_builder2[T, U](const OperatorTerm_t * terms,
                              const BitsetHashMapWrapper& subspace,
                              const T * diag_vec,
                              size_t width,
                              size_t subspace_dim,
                              int has_nonzero_diag,
                              const size_t * group_ptrs,
                              const size_t * group_ladder_ptrs,
                              unsigned int * group_rowint_length,
                              const vector[vector[unsigned int]]& group_offdiag_inds,
                              size_t num_groups,
                              unsigned int ladder_offset,
                              deque[U] csrlike,
                              ) nogil
