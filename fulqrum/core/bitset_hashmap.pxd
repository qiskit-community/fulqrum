from libcpp cimport bool
from .bitset cimport bitset_t

cdef extern from "./src/bitset_hashmap.hpp" namespace "bitset_map_namespace":
    cdef cppclass BitsetHashMapWrapper:
        BitsetHashMapWrapper() except +
        void insert_unique(const bitset_t& bs, size_t value)
        size_t get(const bitset_t& bs)
        bitset_t get_n_th_bitset(size_t n)
        size_t size()
