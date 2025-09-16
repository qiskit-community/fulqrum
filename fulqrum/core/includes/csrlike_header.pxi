# Fulqrum
# Copyright (C) 2024, IBM

from libcpp.vector cimport vector


cdef extern from "../src/csrlike.hpp":
    # CSR like matrix data structures
    ctypedef struct RowData_Real32_t:
        vector[unsigned int] cols
        vector[double] terms


    ctypedef struct RowData_Real64_t:
        vector[size_t] cols
        vector[double] terms
    
    
    ctypedef struct RowData_Complex32_t:
        vector[unsigned int] cols
        vector[double complex] terms


    ctypedef struct RowData_Complex64_t:
        vector[size_t] cols
        vector[double complex] terms
