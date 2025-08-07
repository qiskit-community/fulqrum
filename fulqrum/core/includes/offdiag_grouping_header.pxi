# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
include "fulqrum/core/includes/base_header.pxi"


cdef extern from "../src/offdiag_grouping.hpp":

    size_t term_offdiag_structure(const OperatorTerm_t& term) nogil

    void term_offdiag_sort(vector[OperatorTerm_t]& terms) nogil

    unsigned int _max_offdiag_group_size(size_t * ptrs, size_t num_elems) nogil

    void term_group_sort(vector[OperatorTerm_t]& terms, size_t * weight_ptrs, size_t len_ptrs, 
                                unsigned int max_group_size) nogil
