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

    void set_csr_ptr[T, U](const T& row_data, U * ptrs)

    void set_csr_data[T, U, V](const T& row_data, U * ptrs, U * inds, V * data)

    void dcsrlike_spmv[T, U](const T& row_data, const double * vec, double * out, U dim)

    void zcsrlike_spmv[T, U](const T& row_data, const double complex * vec, double complex * out, U dim)

