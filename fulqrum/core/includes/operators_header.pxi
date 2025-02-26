# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector

include "base_header.pxi"

cdef extern from "../src/operators.hpp":
    void sort_term_data(vector[size_t]& inds, vector[unsigned char]& vals) nogil

    void offdiag_term_sort(QubitOperator_t& oper) nogil
