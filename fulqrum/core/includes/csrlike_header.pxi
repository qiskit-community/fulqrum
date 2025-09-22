# Fulqrum
# Copyright (C) 2024, IBM

from libcpp.vector cimport vector


cdef extern from "../src/csrlike.hpp":
    # CSR like matrix data structures
    ctypedef struct RowData_Real32_t:
        vector[int] cols
        vector[double] data


    ctypedef struct RowData_Real64_t:
        vector[long long] cols
        vector[double] data
    
    
    ctypedef struct RowData_Complex32_t:
        vector[int] cols
        vector[double complex] data


    ctypedef struct RowData_Complex64_t:
        vector[long long] cols
        vector[double complex] data
