# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
include "base_header.pxi"

cdef extern from "../src/grouping.hpp":

    void offdiag_term_sort(QubitOperator_t& oper) nogil


    void compute_term_ladder_inds(const OperatorTerm_t& term, 
                                   unsigned int * ladder_inds, 
                                   unsigned int ladder_width) nogil

    void sort_groups_by_ladder_int(QubitOperator_t& oper,
                                 size_t * group_ptrs,
                                 unsigned int num_groups,
                                 unsigned int ladder_width) nogil

    void ladder_bin_starts(const OperatorTerm_t * terms, const size_t * group_ptrs,
                            unsigned int * group_counts, unsigned int * group_ranges,
                            unsigned int num_groups, unsigned int num_bins, unsigned int ladder_width) nogil

    