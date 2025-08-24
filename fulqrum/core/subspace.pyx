# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.string cimport memcmp
from libc.stdint cimport uint32_t
from libc.math cimport abs
from cython.operator cimport dereference as deref

import math
cimport cython
import numpy as np
cimport numpy as np

from fulqrum.core.subspace cimport Subspace
from fulqrum.core.bitset cimport bitset_t, to_string
from fulqrum.core.bitset_view cimport BitsetView
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper

include "fulqrum/core/includes/base_header.pxi"
include "fulqrum/core/includes/bitset_utils_header.pxi"
include "fulqrum/core/includes/types.pxi"


cdef class Subspace():
    @cython.boundscheck(False)
    def __cinit__(self, dict counts, uint32_t reserve_size, bool full_block=True):
        self.subspace.num_qubits = len(next(iter(counts)))
        self.subspace.size = len(counts)
        if not full_block:
            self.subspace.bitstrings = BitsetHashMapWrapper(full_block)
        # reserve_power_of_2_size = 2 **np.ceil(np.log2(self.subspace.size))
        if reserve_size < self.subspace.size:
            reserve_size = self.subspace.size * 2
        self.subspace.bitstrings.reserve(reserve_size)

        cdef string key
        cdef bitset_t temp_bits
        
        for idx, key in enumerate(counts.keys()):
            temp_bits = bitset_t(key, 0, self.subspace.num_qubits)
            self.subspace.bitstrings.insert_unique(temp_bits, <size_t>idx)
    
    def __dealloc__(self):
        # Clear hash table upon deallocation of class
        self.subspace.bitstrings = BitsetHashMapWrapper()

    @cython.boundscheck(False)
    def __getitem__(self, object key):
        if key < 0:
            key = self.subspace.bitstrings.size() + key 
        cdef size_t idx = <size_t>key
        cdef bitset_t bits = self.subspace.bitstrings.get_n_th_bitset(idx)
        cdef BitsetView view = BitsetView()
        view.assign_bits(bits)
        return view

    def __len__(self):
        return self.subspace.size

    def size(self):
        return self.subspace.size

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
            to_string(self.subspace.bitstrings.get_n_th_bitset(kk), s)
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
            to_string(self.subspace.bitstrings.get_n_th_bitset(kk), s)
            out[s] = None
        return out
