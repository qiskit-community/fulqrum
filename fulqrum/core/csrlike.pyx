# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.vector cimport vector
include "includes/csrlike_header.pxi"
import numpy as np
import scipy.sparse as sp
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
        self.data_d32 = vector[RowData_Real32_t]()
        self.data_d64 = vector[RowData_Real64_t]()
        self.data_z32 = vector[RowData_Complex32_t]()
        self.data_z64 = vector[RowData_Complex64_t]()

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

    def to_csr_array(self):
        cdef int[::1] ptr32
        cdef int[::1] inds32
        cdef long long[::1] ptr64
        cdef long long[::1] inds64
        cdef double[::1] real_data
        cdef complex[::1] complex_data
        
        cdef size_t nnz = self.nnz
        cdef object mat

        if '32' in self.type_string:
            ptr32 = np.zeros(self.num_rows+1, dtype=np.int32)
            inds32 = np.empty(nnz, dtype=np.int32)
        else:
            ptr64 = np.zeros(self.num_rows+1, dtype=np.int64)
            inds64 = np.empty(nnz, dtype=np.int64)

        if 'd' in self.type_string:
            real_data = np.empty(nnz, dtype=float)
        else:
            complex_data = np.empty(nnz, dtype=complex)

        if self.type_string == 'd32':
            set_csr_ptr(self.data_d32, &ptr32[0])
            set_csr_data(self.data_d32, &ptr32[0], &inds32[0], &real_data[0])

            mat = sp.csr_array((real_data, inds32, ptr32), 
                                shape=(self.num_rows,)*2, dtype=float)

        return mat