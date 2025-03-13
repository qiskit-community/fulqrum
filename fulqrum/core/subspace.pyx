# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.string cimport memcmp

import math
cimport cython
import numpy as np
cimport numpy as np

from fulqrumc.core.matrix import FulqrumCSR
from fulqrum.core.spmv cimport FulqrumSpMV

include "fulqrum/core/includes/bitstrings_header.pxi"
include "fulqrum/core/includes/csr_header.pxi"


cdef size_t intmin(size_t a, size_t b):
    return (b-a)*(b < a) + a

cdef size_t MAX_BIN_WIDTH = 26
#Need below to get around Cython issue
ctypedef unsigned char UCHAR

from fulqrum.core.subspace cimport Subspace


cdef class Subspace():
    @cython.boundscheck(False)
    def __cinit__(self, dict counts, size_t bin_width=0):
        self.subspace.num_qubits = len(next(iter(counts)))
        self.subspace.size = len(counts)
        if bin_width == 0:
            bin_width = intmin(<size_t>math.ceil(math.log2(self.subspace.size)), MAX_BIN_WIDTH)
        elif bin_width > self.subspace.num_qubits:
            raise Exception(f'bin_width ({bin_width}) must be <= num_qubits ({self.subspace.num_qubits})')
        elif bin_width > MAX_BIN_WIDTH:
             raise Exception(f'bin_width ({bin_width}) must be <= MAX_BIN_WIDTH ({MAX_BIN_WIDTH})')
            
        self.subspace.bin_width = bin_width
        self.subspace.num_bins = pow(2, bin_width)
        self.subspace.bitstrings.reserve(self.subspace.num_qubits*self.subspace.size)

        # Sort counts according to bin-width
        counts = {k: v for k, v in sorted(counts.items(),
                                          key=lambda item: int(item[0][-bin_width:], 2))}

        cdef size_t kk
        cdef string key
        cdef size_t val
        cdef size_t temp_idx
        cdef size_t bin_idx = 0
        cdef vector[unsigned char] temp_vec
        temp_vec.reserve(self.subspace.num_qubits)
        
        self.subspace.bin_ranges.reserve(self.subspace.num_bins+1)
        self.subspace.bin_counts.reserve(self.subspace.num_bins)
        for kk in range(self.subspace.num_bins):
            self.subspace.bin_counts[kk] = 0     
        
        for key, val in counts.items():
            string_to_vec(key.c_str(), &temp_vec[0], self.subspace.num_qubits)
            for kk in range(self.subspace.num_qubits):
                self.subspace.bitstrings.push_back(temp_vec[kk])
            temp_idx = bin_width_to_int(&temp_vec[0], self.subspace.num_qubits, self.subspace.bin_width)
            self.subspace.bin_counts[temp_idx] += 1

        # Do cumsum to get bin starts and stops
        self.subspace.bin_ranges[0] = 0
        cdef size_t total = self.subspace.bin_counts[0]
        for kk in range(1, self.subspace.num_bins+1):
            self.subspace.bin_ranges[kk] = total
            total += self.subspace.bin_counts[kk]
    
    def __dealloc__(self):
        # Clear vectors upon deallocation of class
        self.subspace.bitstrings = vector[UCHAR]()
        self.subspace.bin_counts = vector[size_t]()
        self.subspace.bin_ranges = vector[size_t]()

    @property
    def bin_width(self):
        return self.subspace.bin_width

    
    @cython.boundscheck(False)
    def __getitem__(self, size_t index):
        cdef unsigned char[::1] out = np.empty(self.subspace.num_qubits, dtype=np.uint8)
        cdef size_t kk
        for kk in range(self.subspace.num_qubits):
            out[kk] = self.subspace.bitstrings[index*self.subspace.num_qubits+kk]
        return np.asarray(out)

    def __len__(self):
        return self.subspace.size

    @cython.boundscheck(False)
    def bin_sizes(self):
        """Array indicating how many vectors are in each bin

        Returns:
            ndarray: Array of type np.uintp
        """
        cdef size_t[::1] out = np.zeros(self.subspace.num_bins, dtype=np.uintp)
        cdef size_t kk
        for kk in range(self.subspace.num_bins):
            out[kk] = self.subspace.bin_counts[kk]
        return np.asarray(out)

    @cython.boundscheck(False)
    def bin_starts(self):
        """Vector indicating start and stop indices for each bin

        Returns:
            ndarray: Array of type np.uintp
        """
        cdef size_t[::1] out = np.zeros(self.subspace.num_bins+1, dtype=np.uintp)
        cdef size_t kk
        for kk in range(self.subspace.num_bins+1):
            out[kk] = self.subspace.bin_ranges[kk]
        return np.asarray(out)

    def vector_bin_index(self, size_t elem):
        """Return the vector at the given index

        Parameters:
            elem (int): Index of element

        Returns:
            ndarray: Array with type np.uintp
        """
        if elem >= self.subspace.size:
            raise Exception(f"Vector index ({elem}) is out of subspace range ({self.subspace.size})")
        cdef int bin_num
        cdef size_t bin_ind, start, stop
        cdef const unsigned char * temp_vec
        temp_vec = &self.subspace.bitstrings[self.subspace.num_qubits*elem]
        bin_num = bin_width_to_int(temp_vec, self.subspace.num_qubits, self.subspace.bin_width)
        start = self.subspace.bin_ranges[bin_num]
        stop = self.subspace.bin_ranges[bin_num+1]
        bin_ind = col_index(start, stop, temp_vec, 
                            &self.subspace.bitstrings[0], self.subspace.num_qubits)
        return (bin_num, bin_ind-start)

    @cython.boundscheck(False)
    def interpret_vector(self, double complex[::1] vec, int sort=0):
        cdef size_t kk, idx
        cdef string temp
        cdef dict out = {}
        cdef unsigned char * sub = &self.subspace.bitstrings[0]
        temp.resize(self.subspace.num_qubits)

        for kk in range(self.subspace.size):
            for idx in range(self.subspace.num_qubits):
                temp[idx] = sub[kk*self.subspace.num_qubits+idx] + 48
            out[temp] = vec[kk]

        if sort:
            out = {k: v for k, v in sorted(out.items(), key=lambda item: int(item[0], 2))}
        return out
    
    @cython.boundscheck(False)
    def to_dict(self):
        cdef size_t kk, idx
        cdef string temp
        cdef dict out = {}
        cdef unsigned char * sub = &self.subspace.bitstrings[0]
        temp.resize(self.subspace.num_qubits)

        for kk in range(self.subspace.size):
            for idx in range(self.subspace.num_qubits):
                temp[idx] = sub[kk*self.subspace.num_qubits+idx] + 48
            out[temp] = None
        return out

    def to_csr(self):
        cdef FulqrumSpMV spmv = self.spmv
        cdef size_t max_int = np.iinfo(np.int32).max
        cdef object out
        cdef size_t num_terms = spmv.oper.terms.size()
        cdef int inds_64 = 0
        if (spmv.subspace_dim > max_int):
            inds_64 = 1
        if num_terms:
            if (spmv.subspace_dim*spmv.oper.terms[num_terms-1].group > max_int):
                inds_64 = 1
        cdef int[::1] indptr32
        cdef int[::1] indices32
        cdef long long[::1] indptr64
        cdef long long[::1] indices64

        if inds_64:
            indptr64 = np.zeros(spmv.subspace_dim+1, dtype=np.int64)
            indices64 = np.zeros(1, dtype=np.int64)
        else:
            indptr32 = np.zeros(spmv.subspace_dim+1, dtype=np.int32)
            indices32 = np.zeros(1, dtype=np.int32)
        
        cdef double complex[::1] data = np.zeros(1, dtype=complex)
        cdef int compute_values = 0
        
        if spmv.diag_vec.shape[0] == 0 and spmv.has_nonzero_diag:
                spmv.compute_diag_vector()
        if inds_64:
            csr_builder[int64](spmv.oper, spmv.subspace.subspace.bitstrings, &spmv.diag_vec[0],
                            spmv.width, spmv.subspace_dim, spmv.has_nonzero_diag,
                            spmv.bin_width, spmv.bin_ranges,
                            &indptr64[0], &indices64[0], &data[0],
                            compute_values)
            if not indptr64[spmv.subspace_dim]:
                return FulqrumCSR(([], [[],[]]), shape=(spmv.subspace_dim,)*2, dtype=complex)
            indices64 = np.zeros(indptr64[spmv.subspace_dim], dtype=np.int64)
            data = np.zeros(indptr64[spmv.subspace_dim], dtype=complex)
        else:
            csr_builder[int](spmv.oper, spmv.subspace.subspace.bitstrings, &spmv.diag_vec[0],
                            spmv.width, spmv.subspace_dim, spmv.has_nonzero_diag,
                            spmv.bin_width, spmv.bin_ranges,
                            &indptr32[0], &indices32[0], &data[0],
                            compute_values)
            if not indptr32[spmv.subspace_dim]:
                return FulqrumCSR(([], [[],[]]), shape=(spmv.subspace_dim,)*2, dtype=complex)
            indices32 = np.zeros(indptr32[spmv.subspace_dim], dtype=np.int32)
            data = np.zeros(indptr32[spmv.subspace_dim], dtype=complex)

        compute_values = 1
        if inds_64:
            csr_builder[int64](spmv.oper, spmv.subspace.subspace.bitstrings, &spmv.diag_vec[0],
                            spmv.width, spmv.subspace_dim, spmv.has_nonzero_diag,
                            spmv.bin_width, spmv.bin_ranges,
                            &indptr64[0], &indices64[0], &data[0],
                            compute_values)
        else:
            csr_builder[int](spmv.oper, spmv.subspace.subspace.bitstrings, &spmv.diag_vec[0],
                            spmv.width, spmv.subspace_dim, spmv.has_nonzero_diag,
                            spmv.bin_width, spmv.bin_ranges,
                            &indptr32[0], &indices32[0], &data[0],
                            compute_values)
        
        if inds_64:
            out = FulqrumCSR((np.asarray(data), np.asarray(indices64), np.asarray(indptr64)),
                            shape=(spmv.subspace_dim,)*2, dtype=complex)
        else:
            out = FulqrumCSR((np.asarray(data), np.asarray(indices32), np.asarray(indptr32)),
                            shape=(spmv.subspace_dim,)*2, dtype=complex)
        # Indices need not be in order, so sort here
        out.sort_indices()
        return out
