# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper
include "base_header.pxi"

cdef extern from "../src/matvec2.hpp":
    void omp_matvec2[T](vector[OperatorTerm_t]& terms,
                const BitsetHashMapWrapper& subspace,
                T * diag_vec,
                size_t width,
                size_t subspace_dim,
                int has_nonzero_diag,
                size_t * group_ptrs,
                size_t * group_ladder_ptrs,
                unsigned int * group_rowint_length,
                const vector[vector[unsigned int]]& group_offdiag_inds,
                unsigned int num_groups,
                unsigned int ladder_offset,
                const T * in_vec,
                T * out_vec) nogil
