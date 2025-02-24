# Fulqrum
# Copyright (C) 2024, IBM

from libcpp.vector cimport vector

cdef extern from "../src/base.hpp":
    ctypedef struct OperatorTerm_t:
        double complex coeff
        vector[size_t] indices
        vector[unsigned char] values
        size_t offdiag_weight
    
    ctypedef struct QubitOperator_t:
        size_t width
        vector[OperatorTerm_t] terms
        int sorted 
