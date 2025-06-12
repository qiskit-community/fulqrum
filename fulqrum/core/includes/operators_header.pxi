# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector

include "base_header.pxi"

cdef extern from "../src/operators.hpp":
    void sort_term_data(vector[unsigned int]& inds, vector[unsigned char]& vals) nogil

    void offdiag_term_sort(QubitOperator_t& oper) nogil

    void set_extended_flag(OperatorTerm_t& term) nogil

    void set_offdiag_weight(OperatorTerm_t& term) nogil

    void combine_qubit_terms(vector[OperatorTerm_t]& terms,
                             vector[OperatorTerm_t]& out_terms,
                             unsigned int * touched,
                             double atol) nogil

    unsigned int term_ladder_int(const OperatorTerm_t& term, unsigned int num_bits) nogil

    void offdiag_weight_sort(QubitOperator_t& oper)
    
    void weight_sort(QubitOperator_t& oper)