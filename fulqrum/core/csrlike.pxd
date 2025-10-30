# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
include "includes/csrlike_header.pxi"
import numpy as np


cdef class CSRLike:
    cdef size_t num_rows
    cdef size_t _nnz
    cdef public unsigned int is_real
    cdef public unsigned int is_int64
    cdef unsigned int data_type
    cdef RowData_Real32_t data_d32
    cdef RowData_Real64_t data_d64
    cdef RowData_Complex32_t data_z32
    cdef RowData_Complex64_t data_z64