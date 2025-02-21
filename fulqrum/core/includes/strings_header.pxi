# Fulqrum
# Copyright (C) 2024, IBM

from libcpp.string cimport string

cdef extern from "../src/strings.hpp":
    string get_column_str(const char * row,
                          size_t bit_len,
                          const size_t * pos,
                          const char * val,
                          size_t N) nogil
