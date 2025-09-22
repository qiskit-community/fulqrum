# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.deque cimport deque
include "includes/csrlike_header.pxi"
import numpy as np
from fulqrum.exceptions import FulqrumError


cdef class CSRLike():
    def __cinit__(self, size_t num_rows, unsigned int is_real=1):
        self.num_rows = num_rows
        self.is_real = is_real
        self.is_int64 = num_rows > np.iinfo(np.int32).max
        self.data_type = 0
        cdef size_t kk
        if self.is_real:
            if self.is_int64:
                self.data_d64.resize(self.num_rows)
                self.data_type = 2
            else:
                self.data_d32.resize(self.num_rows)
                self.data_type = 1
        else:
            if self.is_int64:
                self.data_z64.resize(self.num_rows)
                self.data_type = 4
            else:
                self.data_z32.resize(self.num_rows)
                self.data_type = 3

    def __dealloc__(self):
        # Clear deque upon deallocation of class
        self.data_d32 = deque[RowData_Real32_t]()
        self.data_d64 = deque[RowData_Real64_t]()
        self.data_z32 = deque[RowData_Complex32_t]()
        self.data_z64 = deque[RowData_Complex64_t]()

    @property
    def shape(self):
        return (self.num_rows, self.num_rows)

    @property
    def num_rows(self):
        cdef size_t num_rows = 0
        if self.data_type == 1:
            num_rows = self.data_d32.size()
        elif self.data_type == 2:
            num_rows = self.data_d64.size()
        elif self.data_type == 3:
            num_rows = self.data_z32.size()
        elif self.data_type == 4:
            num_rows = self.data_z64.size()
        return num_rows

    @property
    def type_string(self):
        if self.data_type == 1:
            return 'd32'
        elif self.data_type == 2:
            return 'd64'
        elif self.data_type == 3:
            return 'z32'
        elif self.data_type == 4:
            return 'z64'
        else:
            raise FulqrumError('Invalid data type')

    @property
    def nnz(self):
        cdef size_t nnz = 0
        cdef size_t kk
        if self.data_type == 1:
            for kk in range(self.num_rows):
                nnz += self.data_d32[kk].data.size()
        elif self.data_type == 2:
            for kk in range(self.num_rows):
                nnz += self.data_d64[kk].data.size()
        elif self.data_type == 3:
            for kk in range(self.num_rows):
                nnz += self.data_z32[kk].data.size()
        elif self.data_type == 4:
            for kk in range(self.num_rows):
                nnz += self.data_z64[kk].data.size()
        return nnz

