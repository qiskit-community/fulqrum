# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from fulqrum.core.bitset cimport bitset_t

include "base_header.pxi"


cdef extern from "../src/bitset_utils.hpp":
    
    void bin_int(bitset_t& b, unsigned int bin_width, size_t& res) nogil

    void flip_bits(bitset_t& b, unsigned int * arr, unsigned int size) nogil

    void get_column_bitset(bitset_t& col,
                           unsigned int * pos,
                           unsigned char * val,
                           unsigned int N) nogil

    void sort_bitset_vector(vector[bitset_t]& vec, unsigned int bin_width) nogil

    int nonzero_extended_bitset(const OperatorTerm_t * term,
                                const bitset_t& row) nogil
    


    unsigned int bitset_ladder_int(const bitset_t& row, 
                                   const unsigned int * inds,
                                   unsigned int ladder_width) nogil
