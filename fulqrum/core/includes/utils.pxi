# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector

cdef extern from "../src/utils.hpp":
    void sort_term_data(vector[size_t]& inds, vector[unsigned char]& vals) nogil
