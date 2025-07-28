# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.string cimport memcmp
from libc.math cimport abs
from cython.operator cimport dereference as deref

import math
cimport cython
import numpy as np
cimport numpy as np

from fulqrum.core.subspace cimport Subspace
from fulqrum.core.bitset cimport bitset_t, to_string
from fulqrum.core.bitset_view cimport BitsetView

include "fulqrum/core/includes/base_header.pxi"
include "fulqrum/core/includes/bitset_utils_header.pxi"
include "fulqrum/core/includes/types.pxi"


cdef size_t intmin(size_t a, size_t b):
    return (b-a)*(b < a) + a

cdef size_t MAX_BIN_WIDTH = 32


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
        self.subspace.num_bins = <size_t>(2**bin_width)
        # Number of bitset_t in "bitstrings" is equal to the subspace dimension
        self.subspace.bitstrings.reserve(self.subspace.size)

        # Sort counts according to bin-width
        #counts = {k: v for k, v in sorted(counts.items(),
        #                                  key=lambda item: int(item[0][-bin_width:], 2))}

        cdef size_t kk
        cdef string key
        cdef size_t temp_idx = 0
        cdef size_t bin_idx = 0
        cdef bitset_t temp_bits
        
        self.subspace.bin_ranges.reserve(self.subspace.num_bins+1)
        self.subspace.bin_counts.reserve(self.subspace.num_bins)
        for kk in range(self.subspace.num_bins):
            self.subspace.bin_counts[kk] = 0     
        
        for key in counts.keys():
            temp_bits = bitset_t(key, 0, self.subspace.num_qubits)
            self.subspace.bitstrings.push_back(temp_bits)
            bin_int(temp_bits, bin_width, temp_idx)
            self.subspace.bin_counts[temp_idx] += 1


        sort_bitset_vector(self.subspace.bitstrings, bin_width)

        # Do cumsum to get bin starts and stops
        self.subspace.bin_ranges[0] = 0
        cdef size_t total = self.subspace.bin_counts[0]
        for kk in range(1, self.subspace.num_bins+1):
            self.subspace.bin_ranges[kk] = total
            if kk != self.subspace.num_bins:
                total += self.subspace.bin_counts[kk]
    
    def __dealloc__(self):
        # Clear vectors upon deallocation of class
        self.subspace.bitstrings = vector[bitset_t]()
        self.subspace.bin_counts = vector[size_t]()
        self.subspace.bin_ranges = vector[size_t]()

    @property
    def bin_width(self):
        """Bin-width used in partial sorting of subspace

        Returns:
            int
        """
        return self.subspace.bin_width

    
    @cython.boundscheck(False)
    def __getitem__(self, object key):
        if key < 0:
            key = self.subspace.bitstrings.size() + key 
        cdef size_t idx = <size_t>key
        cdef bitset_t * bits = &self.subspace.bitstrings[idx]
        cdef BitsetView view = BitsetView()
        view.bit_ptr(bits)
        return view

    def __len__(self):
        return self.subspace.size

    def size(self):
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


    @cython.boundscheck(False)
    def interpret_vector(self, double_or_complex[::1] vec, double atol=1e-12, int sort=0):
        """Convert solution vector into dict of counts and complex amplitudes

        Parameters:
            vec (ndarray): Complex solution vector
            atol (double): Absolute tolerance for truncation, default=1e-12
            sort (int): Sort output dict by integer representation.

        Returns:
            dict: Dictionary with bit-string keys and complex values

        Notes:
            Truncation can be disabled by calling `atol=-1`
        """
        cdef size_t kk, idx
        cdef string s
        cdef dict out = {}

        for kk in range(self.subspace.size):
            if abs(vec[kk]) <= atol:
                continue
            to_string(self.subspace.bitstrings[kk], s)
            out[s] = vec[kk]

        if sort:
            out = {k: v for k, v in sorted(out.items(), key=lambda item: int(item[0], 2))}
        return out
    
    @cython.boundscheck(False)
    def to_dict(self):
        """Converts Subspace to a dictionary

        Returns:
            dict
        """
        cdef size_t kk, idx
        cdef string s
        cdef dict out = {}

        for kk in range(self.subspace.size):
            to_string(self.subspace.bitstrings[kk], s)
            out[s] = None
        return out