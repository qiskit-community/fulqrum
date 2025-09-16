# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.deque cimport deque
include "includes/csrlike_header.pxi"
import numpy as np


cdef class CSRLike():
    def __cinit__(self, size_t num_rows, unsigned int is_real=1):
        self.num_rows = num_rows
        self.is_real = is_real
        self.is_int64 = num_rows > np.iinfo(np.int32).max
        self.data_choice = 0
        if self.is_real:
            if self.is_int64:
                self.data_d64.resize(num_rows)
                self.data_choice = 2
            else:
                self.data_d32.resize(num_rows)
                self.data_choice = 1
        else:
            if self.is_int64:
                self.data_z64.resize(num_rows)
                self.data_choice = 4
            else:
                self.data_z32.resize(num_rows)
                self.data_choice = 3

    def __dealloc__(self):
        # Clear deque upon deallocation of class
        self.data_d32 = deque[RowData_Real32_t]()
        self.data_d64 = deque[RowData_Real64_t]()
        self.data_z32 = deque[RowData_Complex32_t]()
        self.data_z64 = deque[RowData_Complex64_t]()

    @property
    def shape(self):
        return (self.num_rows, self.num_rows)
