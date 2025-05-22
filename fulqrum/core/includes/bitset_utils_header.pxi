# Fulqrum
# Copyright (C) 2024, IBM
from fulqrum.core.bitset cimport bitset_t


cdef extern from "../src/bitset_utils.hpp":
    
    void bin_int(bitset_t& b, unsigned int len, size_t& res)

    void flip_bits(bitset_t& b, size_t * arr, size_t size)

    void get_column_bitset(bitset_t& row,
                           bitset_t& col,
                           size_t bit_len,
                           size_t * pos,
                           unsigned char * val,
                           size_t N)
