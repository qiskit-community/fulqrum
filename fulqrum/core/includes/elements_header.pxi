# Fulqrum
# Copyright (C) 2024, IBM

cdef extern from "../src/elements.hpp":

    double complex compute_element_vec(const unsigned char * row,
                                       const unsigned char * col,
                                       size_t bit_len,
                                       const size_t * pos,
                                       const char * val,
                                       double complex coeff,
                                       size_t N) nogil
