# Fulqrum
# Copyright (C) 2024, IBM
from fulqrum.core.bitset cimport bitset_t

cdef extern from "../src/elements.hpp":


    void accum_element(const bitset_t& row,
                       const bitset_t& col,
                       const unsigned int * inds,
                       const unsigned char * val,
                       const double complex& coeff,
                       const unsigned int N,
                       double complex & out)

