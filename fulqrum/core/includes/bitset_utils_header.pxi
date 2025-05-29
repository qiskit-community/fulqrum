# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t


cdef extern from "../src/bitset_utils.hpp":
    
    void bin_int(bitset_t& b, unsigned int bin_width, size_t& res)

    void flip_bits(bitset_t& b, unsigned int * arr, unsigned int size)

    void get_column_bitset(bitset_t& col,
                           unsigned int * pos,
                           unsigned char * val,
                           size_t N)

    void sort_bitset_vector(vector[bitset_t]& vec, unsigned int bin_width)
