# Fulqrum
# Copyright (C) 2024, IBM

from libcpp.vector cimport vector

cdef extern from "../src/csrlike.hpp":
    # CSR like matrix data structures
    ctypedef struct RowData_Real32_t:
        vector[vector[int]] cols
        vector[vector[double]] data


    ctypedef struct RowData_Real64_t:
        vector[vector[long long ]] cols
        vector[vector[double]] data
    
    
    ctypedef struct RowData_Complex32_t:
        vector[vector[int]] cols
        vector[vector[complex]] data


    ctypedef struct RowData_Complex64_t:
        vector[vector[long long ]] cols
        vector[vector[complex]] data

    void set_csr_ptr[T, U](const vector[vector[T]]& cols, U * ptrs)

    void set_csr_data[T, U, V](const vector[vector[T]]& in_data, const vector[vector[U]]& cols, 
                               V * ptrs, V * inds, T * out_data)

    void csrlike_spmv[T, U](const vector[vector[T]]& data, const vector[vector[U]]& cols,
                            const T * vec, T * out, U dim)

    void clear_csrlike_data(vector[vector[int]]& data_d32_cols,
                            vector[vector[double]]& data_d32_data,
                            vector[vector[long long]]& data_d64_cols,
                            vector[vector[double]]& data_d64_data,
                            vector[vector[int]]& data_z32_cols,
                            vector[vector[complex]]& data_z32_data,
                            vector[vector[long long]]& data_z64_cols,
                            vector[vector[complex]]& data_z64_data)
