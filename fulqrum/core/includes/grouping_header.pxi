# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
include "base_header.pxi"

cdef extern from "../src/grouping.hpp":

    void offdiag_term_sort(QubitOperator_t& oper) nogil


    void compute_term_offdiag_inds(const OperatorTerm_t& term, 
                                   unsigned int * offdiag_inds, 
                                   unsigned int num_inds) nogil

    void sort_groups_by_ladder_int(QubitOperator_t& oper,
                                 size_t * group_ptrs,
                                 unsigned int num_groups,
                                 unsigned int ladder_width) nogil

    void ladder_bin_starts(const vector[OperatorTerm_t]& terms, const size_t * group_ptrs,
                            unsigned int * group_counts, size_t * group_ranges,
                            unsigned int num_groups, unsigned int num_bins, unsigned int ladder_width) nogil

    void set_group_offdiag_indices(const vector[OperatorTerm_t]& terms,
                                 vector[vector[unsigned int]]& group_indices,
                                 const size_t * group_ptrs,
                                 unsigned int num_groups) nogil
