# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from libcpp cimport bool
from libc.stdint cimport uint64_t
from .bitset cimport bitset_t

cdef extern from "./src/bitset_hashmap.hpp" namespace "bitset_map_namespace":
    cdef cppclass BitsetHashMapWrapper:
        BitsetHashMapWrapper()
        BitsetHashMapWrapper(bool use_all_bitset_blocks) except +
        void insert_unique(const bitset_t& bs, size_t value)
        void emplace(const bitset_t& bs, size_t value)
        void reserve(const uint64_t num_items)
        size_t get(const bitset_t& bs)
        bitset_t get_n_th_bitset(size_t n)
        size_t size()
        bool use_all_bitset_blocks()
        size_t num_buckets()
