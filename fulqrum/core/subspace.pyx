# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.string cimport memcmp
from libc.stdint cimport uint32_t
from libc.math cimport abs
from cython.operator cimport dereference as deref

import math
cimport cython
import numpy as np
cimport numpy as np

from fulqrum.exceptions import FulqrumError
from fulqrum.core.subspace cimport Subspace
from fulqrum.core.bitset cimport bitset_t, to_string
from fulqrum.core.bitset_view cimport BitsetView
from fulqrum.core.bitset cimport Bitset
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper

include "fulqrum/core/includes/base_header.pxi"
include "fulqrum/core/includes/bitset_utils_header.pxi"
include "fulqrum/core/includes/types.pxi"


cdef class Subspace():
    """Hashmap representation of a quantum subspace over bit-strings obtained from
    sampling on a quantum computer (or simulator).

    Parameters:
        counts (dict): Input counts data formatted as a Python dictionary with bit-strings as strings.
        
        reserve_multiplier (float): We reserve a capacity for the Hash table that stores the
            subspace bit-strings, typically equal to the number of bit-strings. This
            argument allows a user to reserve more capacity than needed. While it consumes,
            more memory, it reduces collision during Hash table look-up leading to 
            minor speed-up.
            Default: 2.
        
        use_all_bitset_blocks (bool): If `use_all_bitset_blocks=False`, only first block of a
            bitset is used in hashing. If your bitsets are long and rarely share common
            prefixes, setting it to False speeds up execution. However, it likely that
            bitsets for practical cases will share common patterns. In that case, set
            `use_all_bitset_blocks` to True so that the whole bitset is used in the hash
            function. Although hashing n (> 1) blocks is slower than hashing a single block,
            full hashing usually leads to fewer collisions during Hash table look-up.
            Default: `True`.
    """
    @cython.boundscheck(False)
    def __cinit__(self, dict counts, int reserve_multiplier=2, bool use_all_bitset_blocks=True):
        self.subspace.num_qubits = len(next(iter(counts)))
        self.subspace.size = len(counts)

        if not use_all_bitset_blocks:
            self.subspace.bitstrings = BitsetHashMapWrapper(use_all_bitset_blocks)
        if reserve_multiplier < 1:
            raise ValueError(
                f"`reserve_multiplier(={reserve_multiplier})` must be >= 1"
            )
        # The +1 is here because insertion would fail for a dim=1 subspace otherwise
        self.subspace.bitstrings.reserve(self.subspace.size * reserve_multiplier + 1)

        cdef size_t idx
        cdef string key
        cdef bitset_t temp_bits
        
        for idx, key in enumerate(counts.keys()):
            temp_bits = bitset_t(key, 0, self.subspace.num_qubits)
            self.subspace.bitstrings.insert_unique(temp_bits, idx)
    
    def __dealloc__(self):
        # Clear hash table upon deallocation of class
        self.subspace.bitstrings = BitsetHashMapWrapper()

    @cython.boundscheck(False)
    def __getitem__(self, object key):
        if key < 0:
            key = self.subspace.bitstrings.size() + key 
        cdef size_t idx = <size_t>key
        cdef bitset_t bits = self.subspace.bitstrings.get_n_th_bitset(idx)
        cdef Bitset out = Bitset()
        out.bits = bits
        return out

    def __len__(self):
        return self.subspace.size

    def size(self):
        return self.subspace.size

    @cython.boundscheck(False)
    def interpret_vector(self, double_or_complex[::1] vec, double atol=1e-12, int sort=0, int renormalize=True):
        """Convert solution vector into dict of counts and complex amplitudes

        Parameters:
            vec (ndarray): Complex solution vector
            atol (double): Absolute tolerance for truncation, default=1e-12
            sort (int): Sort output dict by integer representation.
            renormalize (bool): Renormalize values such that probabilities sum to one, default = True

        Returns:
            dict: Dictionary with bit-string keys and complex values

        Notes:
            Truncation can be disabled by calling `atol=0`
        """
        cdef size_t kk, idx
        cdef double abs_val
        cdef double reduced_prob = 0
        cdef string s
        cdef dict out = {}

        for kk in range(self.subspace.size):
            abs_val = abs(vec[kk])
            if abs_val <= atol:
                continue
            to_string(self.subspace.bitstrings.get_n_th_bitset(kk), s)
            out[s] = vec[kk]
            reduced_prob += abs_val * abs_val

        if renormalize:
            reduced_prob = math.sqrt(reduced_prob)
            for key in out:
                out[key] /= reduced_prob

        if sort:
            out = {k: v for k, v in sorted(out.items(), key=lambda item: int(item[0], 2))}
        return out
    
    def get_n_th_bitstring(self, size_t n):
        """Return n-th bitstring in the Subspace

        Parameters:
            n (size_t): Index of the expected bitstring.

        Returns:
            str: N-th bitstring in the subspace. Note that, both Python
                dictionary and emhash8::HashMap retains the insertion order.
        """
        cdef string s
        to_string(self.subspace.bitstrings.get_n_th_bitset(n), s)
        return s

    def get_bitstring_index(self, Bitset bitstring):
        """Return the index of the given bitstring.

        Return value is max(size_t) if bitstring not in subspace

        Returns:
            size_t: Index
        """
        return self.subspace.bitstrings.get(bitstring.bits)
    
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
