# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.vector cimport vector
include "includes/csrlike_header.pxi"
include "includes/types.pxi"
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
                self.data_d64.cols.resize(self.num_rows)
                self.data_d64.data.resize(self.num_rows)
                self.data_type = 2
            else:
                self.data_d32.cols.resize(self.num_rows)
                self.data_d32.data.resize(self.num_rows)
                self.data_type = 1
        else:
            if self.is_int64:
                self.data_z64.cols.resize(self.num_rows)
                self.data_z64.data.resize(self.num_rows)
                self.data_type = 4
            else:
                self.data_z32.cols.resize(self.num_rows)
                self.data_z32.data.resize(self.num_rows)
                self.data_type = 3

    def __dealloc__(self):
        # Clear cols and data vectors upon deallocation of class
        self.data_d32.cols = vector[vector[int]]()
        self.data_d64.cols = vector[vector[int64]]()
        self.data_z32.cols = vector[vector[int]]()
        self.data_z64.cols = vector[vector[int64]]()
        
        self.data_d32.data = vector[vector[double]]()
        self.data_d32.data = vector[vector[double]]()
        self.data_z32.data = vector[vector[complex]]()
        self.data_z64.data = vector[vector[complex]]()

    @property
    def shape(self):
        return (self.num_rows, self.num_rows)

    @property
    def num_rows(self):
        cdef size_t num_rows = 0
        if self.data_type == 1:
            num_rows = self.data_d32.cols.size()
        elif self.data_type == 2:
            num_rows = self.data_d64.cols.size()
        elif self.data_type == 3:
            num_rows = self.data_z32.cols.size()
        elif self.data_type == 4:
            num_rows = self.data_z64.cols.size()
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
                nnz += self.data_d32.data[kk].size()
        elif self.data_type == 2:
            for kk in range(self.num_rows):
                nnz += self.data_d64.data[kk].size()
        elif self.data_type == 3:
            for kk in range(self.num_rows):
                nnz += self.data_z32.data[kk].size()
        elif self.data_type == 4:
            for kk in range(self.num_rows):
                nnz += self.data_z64.data[kk].size()
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
            set_csr_ptr(self.data_d32.cols, &ptr32[0])
            set_csr_data(self.data_d32.data, self.data_d32.cols, &ptr32[0], &inds32[0], &real_data[0])

            mat = sp.csr_array((real_data, inds32, ptr32), 
                                shape=(self.num_rows,)*2, dtype=float)

        elif self.type_string == 'd64':
            set_csr_ptr(self.data_d64.cols, &ptr64[0])
            set_csr_data(self.data_d64.data, self.data_d64.cols, &ptr64[0], &inds64[0], &real_data[0])

            mat = sp.csr_array((real_data, inds64, ptr64), 
                                shape=(self.num_rows,)*2, dtype=float)

        elif self.type_string == 'z32':
            set_csr_ptr(self.data_z32.cols, &ptr32[0])
            set_csr_data(self.data_z32.data, self.data_z32.cols, &ptr32[0], &inds32[0], &complex_data[0])

            mat = sp.csr_array((complex_data, inds32, ptr32), 
                                shape=(self.num_rows,)*2, dtype=complex)
        
        elif self.type_string == 'z64':
            set_csr_ptr(self.data_z64.cols, &ptr64[0])
            set_csr_data(self.data_z64.data, self.data_z64.cols, &ptr64[0], &inds64[0], &complex_data[0])

            mat = sp.csr_array((complex_data, inds64, ptr64), 
                                shape=(self.num_rows,)*2, dtype=complex)

        return mat

    def matvec(self,  double_or_complex[::1] x):
        if <size_t>x.shape[0] != self.num_rows:
            raise FulqrumError('Incorrect length of input vector.')

        cdef double_or_complex[::1] out
        if self.is_real:
            out = np.zeros(x.shape[0], dtype=float)
        else:
            out = np.zeros(x.shape[0], dtype=complex)

        if self.type_string == 'd32':
            if double_or_complex is double: #This is here to allow for us to specialize type
                csrlike_spmv(self.data_d32.data, self.data_d32.cols, &x[0], &out[0], <int>self.num_rows)

        elif self.type_string == 'd64':
            if double_or_complex is double:
                csrlike_spmv(self.data_d64.data, self.data_d64.cols, &x[0], &out[0], <long long>self.num_rows)

        if self.type_string == 'z32':
            if double_or_complex is complex:
                csrlike_spmv(self.data_z32.data, self.data_z32.cols, &x[0], &out[0], <int>self.num_rows)

        elif self.type_string == 'z64':
            if double_or_complex is complex:
                csrlike_spmv(self.data_z64.data, self.data_z64.cols, &x[0], &out[0], <long long>self.num_rows)
        
        return np.asarray(out)