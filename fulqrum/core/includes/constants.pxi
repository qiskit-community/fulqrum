# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cdef double RTOL = 1e-8
cdef double ATOL = 1e-14

cdef extern from "../src/constants.hpp":
    cdef size_t MAX_SIZE_T = MAX_SIZE_T
