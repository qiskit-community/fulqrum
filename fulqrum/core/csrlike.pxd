# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
include "includes/csrlike_header.pxi"
from libcpp.deque cimport deque
import numpy as np


cdef class CSRLike:
    cdef size_t num_rows
    cdef public unsigned int is_real
    cdef public unsigned int is_int64
    cdef unsigned int data_choice
    cdef deque[RowData_Real32_t] data_d32
    cdef deque[RowData_Real64_t] data_d64
    cdef deque[RowData_Complex32_t] data_z32
    cdef deque[RowData_Complex64_t] data_z64