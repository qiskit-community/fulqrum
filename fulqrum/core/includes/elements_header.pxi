# Fulqrum
# Copyright (C) 2024, IBM

cdef extern from "../src/elements.hpp":

    void accum_element_value(const unsigned char * row,
                             const unsigned char * col,
                             const size_t bit_len,
                             const unsigned int * pos,
                             const unsigned char * val,
                             const double complex coeff,
                             const size_t N,
                             double complex& out) nogil
