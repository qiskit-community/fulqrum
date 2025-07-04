# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
include "base_header.pxi"

cdef extern from "../src/grouping.hpp":

    void offdiag_term_sort(QubitOperator_t& oper) nogil


    void compute_term_ladder_inds(const OperatorTerm_t& term, 
                                   vector[unsigned int]& ladder_inds, 
                                   unsigned int ladder_width) nogil

    