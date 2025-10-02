# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t
from libcpp cimport bool
from fulqrum.core.bitset cimport bitset_t

include "base_header.pxi"


cdef extern from "../src/bitset_utils.hpp":

    void flip_bits(bitset_t& b, const unsigned int * arr, const unsigned int size) nogil

    void get_column_bitset(bitset_t& col,
                           const unsigned int * pos,
                           const unsigned char * val,
                           const unsigned int N) nogil

    unsigned int bitset_ladder_int(const uint8_t * row, 
                                   const unsigned int * inds,
                                   const unsigned int ladder_width) nogil

    unsigned int passes_proj_validation(const OperatorTerm_t * term,
                                        const bitset_t& row) nogil
