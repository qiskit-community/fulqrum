from libcpp cimport bool
from libc.stdint cimport uint64_t
from .bitset cimport bitset_t

cdef extern from "./src/bitset_hashmap.hpp" namespace "bitset_map_namespace":
    cdef cppclass BitsetHashMapWrapper:
        BitsetHashMapWrapper()
        BitsetHashMapWrapper(bool use_all_bitset_blocks) except +
        void insert_unique(const bitset_t& bs, size_t value)
        void reserve(const uint64_t num_items)
        size_t get(const bitset_t& bs)
        bitset_t get_n_th_bitset(size_t n)
        size_t size()
