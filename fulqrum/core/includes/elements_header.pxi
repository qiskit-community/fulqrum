# Fulqrum
# Copyright (C) 2024, IBM
from fulqrum.core.bitset cimport bitset_t

cdef extern from "../src/elements.hpp":

    void accum_element_value(const unsigned char * row,
                             const unsigned char * col,
                             const unsigned int bit_len,
                             const unsigned int * pos,
                             const unsigned char * val,
                             const double complex coeff,
                             const size_t N,
                             double complex& out)


    void accum_element(const bitset_t& row,
                       const bitset_t& col,
                       const unsigned int * inds,
                       const unsigned char * val,
                       const double complex& coeff,
                       const unsigned int N,
                       double complex & out)

